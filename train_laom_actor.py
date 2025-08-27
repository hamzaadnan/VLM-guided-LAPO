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

from copy import deepcopy
from PIL import Image
from tqdm import trange
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataclasses import asdict, field, dataclass
from torchvision.utils import make_grid

from src.nn import LAOM, ActionDecoder, ObsActionDecoder, Actor
from src.scheduler import linear_annealing_with_warmup
from src.data_classes import LAOMConfig, BCConfig, DecoderConfig
from src.utils import (
    DCSChunkedDataset,
    DCSChunkedLAOMDataset,
    soft_update,
    worker_init_fn,
    create_env_from_df,
    get_grad_norm,
    get_optim_groups,
    normalise_img,
    unnormalise_img,
    set_seed
)

@dataclass
class Config:
    project: str = "weighted_lapo"
    group: str = "lapo_weighted"
    name: str = "lapo_weighted"
    seed: int = 0
    device: str = "cuda:0"
    lapo: LAOMConfig = field(default_factory=LAOMConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # def __post_init__(self):
    #     self.name = f"{self.name}-{str(uuid.uuid4())}"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train_lapo(config: LAOMConfig, DEVICE: str) -> LAOM:
    dataset = DCSChunkedLAOMDataset(
        hdf5_path=config.data_path,
        frame_stack=config.frame_stack,
        max_offset=config.frame_stack,
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

    lapo = LAOM(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw), # type: ignore
        latent_act_dim=config.latent_action_dim,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
        encoder_norm_out=config.encoder_norm_out,
        act_head_dim=config.latent_action_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.latent_action_dim,
        obs_head_dropout=config.obs_head_dropout
    ).to(DEVICE)
    
    target_lapo = deepcopy(lapo)

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

    state_probe = nn.Linear(
        math.prod(lapo.final_encoder_shape),
        dataset.state_dim
    ).to(DEVICE)
    state_probe_optim = torch.optim.Adam(
        state_probe.parameters(),
        lr=config.learning_rate
    )

    act_linear_probe = nn.Linear(
        config.latent_action_dim, 
        dataset.act_dim
    ).to(DEVICE)
    act_probe_optim = torch.optim.Adam(
        act_linear_probe.parameters(),
        lr=config.learning_rate
    )
    
    state_act_linear_probe = nn.Linear(
        math.prod(lapo.final_encoder_shape),
        dataset.act_dim
    ).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(
        state_act_linear_probe.parameters(),
        lr=config.learning_rate
    )

    # Scheduler
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(
        optimiser=optim,
        warmup_steps=warmup_updates,
        total_steps=total_updates
    )


    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
        for idx, batch in enumerate(dataloader):
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, future_obs, states, actions, _ = [b.to(DEVICE) for b in batch]
            obs = normalise_img(obs) # shape: [B, F * 3, H, W]
            next_obs = normalise_img(next_obs)
            future_obs = normalise_img(future_obs)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent_next_obs, latent_action, obs_hidden = lapo(obs, future_obs)

                with torch.no_grad():
                    next_obs_target = target_lapo.encoder(next_obs).flatten(1)

                loss = F.mse_loss(
                    latent_next_obs,
                    next_obs_target.detach()
                )
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            if idx % config.target_update_every == 0:
                soft_update(
                    target_lapo,
                    lapo,
                    tau=config.target_tau
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, states)
            
            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_action = act_linear_probe(latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action, actions)

            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                state_pred_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(state_pred_action, actions)
            
            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()

            wandb.log(
                {
                    "lapo/mse_loss": loss.item(),
                    "lapo/state_probe_mse_loss": state_probe_loss.item(),
                    "lapo/action_probe_mse_loss": act_probe_loss.item(),
                    "lapo/state_action_probe_mse_loss": state_act_probe_loss.item(),
                    "lapo/throughput": total_tokens / (time.time() - start_time),
                    "lapo/learning_rate": scheduler.get_last_lr()[0],
                    "lapo/grad_norm": get_grad_norm(lapo),
                    "lapo/target_obs_norm": torch.norm(next_obs_target, p=2, dim=-1).mean().item(),
                    "lapo/online_obs_norm": torch.norm(latent_next_obs, p=2, dim=-1).mean().item(),
                    "lapo/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                    "lapo/epoch": epoch,
                    "lapo/total_steps": total_steps
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

def train_bc(lam: LAOM, config: BCConfig, DEVICE: str) -> Actor:
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
    torch.save(lapo.state_dict(), f"saved_models/lams/l-{config.name}.pth")

    actor = train_bc(lam=lapo, config=config.bc, DEVICE=config.device)
    torch.save(actor.state_dict(), f"saved_models/actors/l-{config.name}.pth")

    config_path = f"saved_models/configs/l-{config.name}.yaml"
    with open(config_path, "w") as f:
       yaml.dump(
           asdict(config), f, default_flow_style=False, sort_keys=False # type: ignore
       ) 

    _ = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc, DEVICE=config.device)

    run.finish()


if __name__ == "__main__":
    main() # type: ignore