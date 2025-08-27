import uuid
from dataclasses import dataclass, field

@dataclass
class LAPOConfig:
    num_epochs: int = 10
    batch_size: int = 512
    future_obs_offset: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 3
    latent_action_dim: int = 16
    encoder_scale: int = 6
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    weighted: bool = False
    frame_stack: int = 3
    data_path: str = "/data/lynx/ms24ha/64px_data/hopper-hop-scale-easy-video-hard-64px-5k.hdf5"

@dataclass
class LAOMConfig:
    num_epochs: int = 10
    batch_size: int = 512
    future_obs_offset: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 3
    latent_action_dim: int = 16
    act_head_dropout: float = 0.0
    obs_head_dropout: float = 0.0
    encoder_scale: int = 0
    encoder_num_res_blocks: int = 2
    encoder_dropout: float = 0.0
    encoder_norm_out: bool = False
    encoder_deep: bool = False
    target_tau: float = 1e-3
    target_update_every: int = 1
    frame_stack: int = 3
    data_path: str = "/data/lynx/ms24ha/64px_data/hopper-hop-scale-easy-video-hard-64px-5k.hdf5"

@dataclass
class BCConfig:
    num_epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0 
    warmup_epochs: int = 0
    encoder_scale: int = 8
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    frame_stack: int = 3
    data_path: str = "/data/lynx/ms24ha/64px_data/hopper-hop-scale-easy-video-hard-64px-5k.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 5
    eval_seed: int = 0

@dataclass
class DecoderConfig:
    use_obs: bool = False
    load_actor: bool = False
    actor_path: str = "/data/tucana/ms24ha/compr_recon/saved_models/actors/lapo_weighted-2240f441-d055-4e7d-905c-1765e449def5.pth"
    total_updates: int = 2500
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 0
    hidden_dim: int = 256
    data_path: str = "/data/lynx/ms24ha/64px_data/hopper-hop-scale-easy-video-hard-labeled-1000x2-64px.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 25
    eval_seed: int = 0

@dataclass
class Config:
    project: str = "weighted_lapo"
    group: str = "lapo_weighted"
    name: str = "lapo_weighted"
    seed: int = 0
    device: str = "cuda:0"
    lapo: LAPOConfig = field(default_factory=LAPOConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"

