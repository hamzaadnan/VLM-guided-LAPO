import math
import uuid
import time
import yaml
import torch
import torchinfo
import pyrallis 
import wandb
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from PIL import Image
from tqdm import trange
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataclasses import dataclass, asdict, field
from torchvision.utils import make_grid

from src.nn import LAPOLabels, ActionDecoder, ObsActionDecoder, Actor
from src.scheduler import linear_annealing_with_warmup
from src.data_classes import LAPOLabelConfig, BCConfig, DecoderConfig
from src.utils import (
    DCSChunkedDataset,
    DCSChunkedHeatmapDataset,
    DCSChunkedIterableDataset,
    weighted_mse_loss,
    worker_init_fn,
    create_env_from_df,
    get_grad_norm,
    get_optim_groups,
    normalise_img,
    unnormalise_img,
    set_seed
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class Config:
    project: str = "weighted_lapo"
    group: str = "lapo_weighted"
    name: str = "lapo_weighted"
    seed: int = 0
    device: str = "cuda:0"
    lapo: LAPOLabelConfig = field(default_factory=LAPOLabelConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


def train_lapo(config: LAPOLabelConfig, DEVICE: str) -> LAPOLabels:
    dataset = DCSChunkedHeatmapDataset(
        hdf5_path=config.data_path,
        frame_stack=config.frame_stack,
        max_offset=config.frame_stack,
    )
    labelled_dataset = DCSChunkedIterableDataset(
        hdf5_path=config.labelled_data_path,
        frame_stack=config.frame_stack,
        max_offset=config.future_obs_offset
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )
    labelled_dataloader = DataLoader(
        dataset=labelled_dataset,
        batch_size=config.labelled_batch_size,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        # pin_memory=True,
        drop_last=True,
    )

    lapo = LAPOLabels(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        latent_act_dim=config.latent_action_dim,
        true_act_dim=dataset.act_dim,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks
    ).to(DEVICE)

    if config.weighted:
        print("Using weighted loss to train LAM.")

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw)
        ]
    )

    optim = torch.optim.Adam(
        params=get_optim_groups(lapo, config.weight_decay),
        lr=config.learning_rate,
        fused=True
    )
    
    # Scheduler
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(
        optimiser=optim,
        warmup_steps=warmup_updates,
        total_steps=total_updates
    )

    linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    probe_optim = torch.optim.Adam(linear_probe.parameters(), lr=config.learning_rate)

    labeled_dataloader_iter = iter(labelled_dataloader)


    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, future_obs, heatmaps, actions, _ = [b.to(DEVICE) for b in batch]
            obs = normalise_img(obs) # shape: [B, F * 3, H, W]
            next_obs = normalise_img(next_obs)
            future_obs = normalise_img(future_obs)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_next_obs, latent_action = lapo(obs, future_obs, pred_true_act=False)
                if config.weighted:
                    recon_loss = weighted_mse_loss(pred_next_obs, next_obs, heatmaps) # type: ignore
                else:
                    recon_loss = F.mse_loss(pred_next_obs, next_obs) # type: ignore

            labelled_batch = next(labeled_dataloader_iter)
            labelled_obs, labelled_next_obs, labelled_future_obs, labelled_actions, _ = [b.to(DEVICE) for b in labelled_batch]

            labelled_obs = normalise_img(labelled_obs)
            labelled_next_obs = normalise_img(labelled_next_obs)
            labelled_future_obs = normalise_img(labelled_future_obs)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, pred_actions = lapo(labelled_obs, labelled_future_obs, pred_true_act=True)
                true_action_loss = F.mse_loss(pred_actions, labelled_actions)      
      
            loss = recon_loss + config.labelled_loss_coeff * true_action_loss            

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_action = linear_probe(latent_action.detach())
                probe_loss = F.mse_loss(pred_action, actions)

            probe_optim.zero_grad(set_to_none=True)
            probe_loss.backward()
            probe_optim.step()

            wandb.log(
                {
                    "lapo/mse_loss": loss.item(),
                    "lapo/action_probe_mse_loss": probe_loss.item(),
                    "lapo/throughput": total_tokens / (time.time() - start_time),
                    "lapo/learning_rate": scheduler.get_last_lr()[0],
                    "lapo/grad_norm": get_grad_norm(lapo),
                    "lapo/epoch": epoch,
                    "lapo/total_steps": total_steps
                }
            )
        
        obs_example = [unnormalise_img(next_obs[0, i : i + 3]) for i in range(0, 3 * config.frame_stack, 3)] # type: ignore
        next_obs_example = [unnormalise_img(pred_next_obs[0, i : i + 3]) for i in range(0, 3 * config.frame_stack, 3)] # type: ignore
        reconstruction_img = make_grid(obs_example + next_obs_example, nrow=config.frame_stack, padding=1) # type: ignore
        reconstruction_img = Image.fromarray(
            reconstruction_img.permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
        )
        reconstruction_img = wandb.Image(reconstruction_img, caption="Top: True, Bottom: Predicted")

        loss_diff = [
            unnormalise_img(((pred_next_obs[0, i : i + 3] - next_obs[0, i : i + 3]) ** 2)) for i in range(0, 3 * config.frame_stack, 3) # type: ignore
        ] 
        loss_diff = make_grid(loss_diff, nrow=config.frame_stack, padding=1)
        loss_diff = Image.fromarray(
            loss_diff.permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
        )
        loss_diff = wandb.Image(loss_diff, caption="Spatial Loss Difference")
        wandb.log(
            {
                "lapo/next_obs_pred": reconstruction_img,
                "lapo/loss_diff": loss_diff,
                "lapo/epoch": epoch,
                "lapo/total_steps": total_tokens
            }
        )
    return lapo

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

def train_bc(lam: LAPOLabels, config: BCConfig, DEVICE: str) -> Actor:
    dataset = DCSChunkedDataset(
        hdf5_path=config.data_path,
        frame_stack=config.frame_stack,
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
    eval_env = create_env_from_df(
        hdf5_path=config.data_path,
        backgrounds_path=config.dcs_backgrounds_path,
        frame_stack=config.frame_stack,
        backgrounds_split=config.dcs_backgrounds_split
    )
    print(f"Evaluation Env Observation Space: {eval_env.observation_space}")
    print(f"Evaluation Env Action Space: {eval_env.action_space}")

    num_actions = lam.latent_act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor = Actor(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        params=get_optim_groups(actor, config.weight_decay),
        lr=config.learning_rate,
        fused=True
    )

    # Scheduler
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(
        optimiser=optim,
        warmup_steps=warmup_updates,
        total_steps=total_updates
    )

    # Action Decoder
    print(f"Latent Action Dimension = {num_actions}")
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 16),
        nn.ReLU6(),
        nn.Linear(16, 16),
        nn.ReLU6(),
        nn.Linear(16, dataset.act_dim)
    ).to(DEVICE)
    act_decoder_optim = torch.optim.AdamW(
        params=act_decoder.parameters(),
        lr=config.learning_rate,
        fused=True
    )
    act_decoder_scheduler = linear_annealing_with_warmup(
        optimiser=act_decoder_optim,
        warmup_steps=warmup_updates, total_steps=total_updates
    )

    torchinfo.summary(
        actor,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw)
        ]
    )
    
    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, true_actions = [b.to(DEVICE) for b in batch]
            obs = normalise_img(obs)
            next_obs = normalise_img(next_obs)
            target_actions = lam.label(obs, next_obs)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_actions, _ = actor(obs)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                decoder_loss = F.mse_loss(pred_true_actions, true_actions)
            
            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            wandb.log(
                {
                    "bc/mse_loss": loss.item(),
                    "bc/throughput": total_tokens / (time.time() - start_time),
                    "bc/learning_rate": scheduler.get_last_lr()[0],
                    "bc/act_decoder_probe_mse_loss": decoder_loss.item(),
                    "bc/epoch": epoch,
                    "bc/total_steps": total_steps
                }
            )
        
    actor.eval()
    eval_returns = evaluate_bc(
        env=eval_env,
        actor=actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=act_decoder
    )

    wandb.log(
        {
            "bc/eval_returns_mean": eval_returns.mean(),
            "bc/eval_returns/std": eval_returns.std(),
            "bc/total_steps": total_steps
        }
    )

    return actor

def train_act_decoder(actor: Actor, config: DecoderConfig, bc_config: BCConfig, DEVICE: str) -> ActionDecoder | ObsActionDecoder:
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
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True
    )
    set_seed(config.seed)

    lapo = train_lapo(config=config.lapo, DEVICE=config.device)
    torch.save(lapo.state_dict(), f"saved_models/lams/labelled-{config.name}.pth")

    actor = train_bc(lam=lapo, config=config.bc, DEVICE=config.device)
    torch.save(actor.state_dict(), f"saved_models/actors/labelled-{config.name}.pth")

    config_path = f"saved_models/configs/labelled-{config.name}.yaml"
    with open(config_path, "w") as f:
       yaml.dump(
           asdict(config), f, default_flow_style=False, sort_keys=False # type: ignore
       ) 

    _ = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc, DEVICE=config.device)

    run.finish()


if __name__ == "__main__":
    main() # type: ignore