import math
import time
import torch
import pyrallis
import wandb
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import asdict
from tqdm import trange
from torch.utils.data import DataLoader

from src.nn import ActionDecoder, ObsActionDecoder, Actor
from src.scheduler import linear_annealing_with_warmup
from src.data_classes import DecoderConfig, BCConfig, Config
from src.utils import (
    DCSChunkedDataset,
    worker_init_fn,
    create_env_from_df,
    get_optim_groups,
    normalise_img,
    set_seed
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def evaluate_bc(
        env: gym.Env,
        actor: Actor,
        num_episodes: int,
        seed: int = 0,
        device: str = "cpu",
        action_decoder: nn.Module | None = None
    ) -> np.ndarray:
    returns = []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        total_reward = 0.0
        obs, _ = env.reset(seed = seed + ep)
        done = False
        while not done:
            obs_ = torch.tensor(obs.copy(), device=device)[None] # shape: [1, 64, 64, F*C]
            obs_ = obs_.permute((0, 3, 1, 2))
            obs_ = normalise_img(obs_)
            action, obs_emb = actor(obs_)
            if action_decoder is not None:
                if isinstance(action_decoder, ObsActionDecoder):
                    action = action_decoder(action, obs_emb)
                else:
                    action = action_decoder(action)
            obs, reward, terminated, truncated, _ = env.step(
                action.squeeze().cpu().numpy()
            )
            done = terminated or truncated
            total_reward += float(reward) 
        returns.append(total_reward)
    return np.array(returns)


def train_act_decoder(actor: Actor, config: DecoderConfig, bc_config: BCConfig, DEVICE: str) -> nn.Module:
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    dataset = DCSChunkedDataset(
        hdf5_path=config.data_path,
        frame_stack=bc_config.frame_stack
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    num_epochs = config.total_updates // len(dataloader)

    if config.use_obs:
        action_decoder = ObsActionDecoder(
            latent_act_dim=actor.num_actions,
            obs_emb_dim=actor.final_encoder_shape[0],
            true_act_dim=dataset.act_dim,
            hidden_dim=config.hidden_dim
        ).to(DEVICE)
    else:
        action_decoder = ActionDecoder(
            latent_act_dim=actor.num_actions,
            true_act_dim=dataset.act_dim,
            hidden_dim=config.hidden_dim
        ).to(DEVICE)

    optim = torch.optim.AdamW(
        params=get_optim_groups(action_decoder, config.weight_decay),
        lr=config.learning_rate,
        fused=True
    )
    eval_env = create_env_from_df(
        hdf5_path=config.data_path,
        backgrounds_path=config.dcs_backgrounds_path,
        backgrounds_split=config.dcs_backgrounds_split,
        frame_stack=bc_config.frame_stack
    )
    print(f"Evaluation Env Observation Space: {eval_env.observation_space}")
    print(f"Evaluation Env Action Space: {eval_env.action_space}")

    # Scheduler
    total_updates = len(dataloader) * num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(
        optimiser=optim,
        warmup_steps=warmup_updates,
        total_steps=total_updates
    )

    
    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(num_epochs, desc="Epochs"):
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, _, true_actions = [b.to(DEVICE) for b in batch]
            obs = normalise_img(obs)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    latent_actions, obs_emb = actor(obs)
                if config.use_obs:
                    pred_actions = action_decoder(latent_actions, obs_emb)
                else:
                    pred_actions = action_decoder(latent_actions)
                loss = F.mse_loss(pred_actions, true_actions)
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            wandb.log(
                {
                    "decoder/mse_loss": loss.item(),
                    "decoder/throughput": total_tokens / (time.time() - start_time),
                    "decoder/learning_rate": scheduler.get_last_lr()[0],
                    "decoder/epoch": epoch,
                    "decoder/total_steps": total_steps
                }
            )
    actor.eval()
    eval_returns = evaluate_bc(
        env=eval_env,
        actor=actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=action_decoder
    )

    wandb.log(
        {
            "decoder/eval_returns_mean": eval_returns.mean(),
            "decoder/eval_returns_std": eval_returns.std(),
            "decoder/total_steps": total_steps
        }
    )

    return action_decoder


@pyrallis.wrap() # type: ignore
def main(config: Config):
    assert config.decoder.load_actor == True, "Actor must be provided"
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True
    )
    # run.config["seed"] = wandb.config["seed"]
    # run.config["decoder"]["use_obs"] = wandb.config["decoder_use_obs"]
    # run.config["decoder"]["data_path"] = wandb.config["decoder_data_path"]

    set_seed(config.seed)

    actor = Actor(
        shape=(3 * config.bc.frame_stack, 64, 64),
        num_actions=config.lapo.latent_action_dim,
        encoder_scale=config.bc.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.bc.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.bc.encoder_num_res_blocks,
        dropout=config.bc.dropout
    ).to(config.device)
    checkpoint = torch.load(
        config.decoder.actor_path,
        map_location=config.device
    )
    actor.load_state_dict(checkpoint)
    
    _ = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc, DEVICE=config.device)

    run.finish()

if __name__ == "__main__":
    main() # type: ignore