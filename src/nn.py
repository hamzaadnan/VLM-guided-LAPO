import math
import torch
import torch.nn as nn
from .utils import weight_init

class MLPBlock(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand * dim),
            nn.ReLU6(),
            nn.Linear(expand * dim, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))
    
# Projects from an embedding space to the action space
class LatentActHead(nn.Module):
    def __init__(self, act_dim, emb_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        self.proj0 = nn.Linear(2 * emb_dim, hidden_dim)
        self.proj1 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, act_dim)
        
        self.block0 = MLPBlock(dim=hidden_dim, expand=expand, dropout=dropout)
        self.block1 = MLPBlock(dim=hidden_dim, expand=expand, dropout=dropout)
        self.block2 = MLPBlock(dim=hidden_dim, expand=expand, dropout=dropout)

    def forward(self, obs_emb, next_obs_emb):
        x = self.block0(
            self.proj0(
                torch.concat([obs_emb, next_obs_emb], dim=-1)
            )
        )
        x = self.block1(
            self.proj1(
                torch.concat([x, obs_emb, next_obs_emb], dim=-1)
            )
        )
        x = self.block2(
            self.proj2(
                torch.concat([x, obs_emb, next_obs_emb], dim=-1)
            )
        )
        return self.proj_end(x)
    

# This takes a latent observation and latent action and predicts the next latent observation
class LatentObsHead(nn.Module):
    def __init__(self, act_dim, proj_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        self.proj0 = nn.Linear(act_dim + proj_dim, hidden_dim)
        self.proj1 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, proj_dim)

        self.block0 = MLPBlock(dim=hidden_dim, expand=expand, dropout=dropout)
        self.block1 = MLPBlock(dim=hidden_dim, expand=expand, dropout=dropout)
        self.block2 = MLPBlock(dim=hidden_dim, expand=expand, dropout=dropout)

    def forward(self, x, action):
        x = self.block0(
            self.proj0(
                torch.concat([x, action], dim=-1)
            )
        )
        x = self.block1(
            self.proj1(
                torch.concat([x, action], dim=-1)
            )
        )
        x = self.block2(
            self.proj2(
                torch.concat([x, action], dim=-1)
            )
        )
        return self.proj_end(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2, dropout=0.0, downscale=True):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._downscale = downscale
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self._downscale else 1,
        )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels, dropout) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.blocks(self.conv(x))
        assert x.shape[1:] == self.get_output_shape()
        return x
    
    def get_output_shape(self):
        _C, H, W = self._input_shape
        if self._downscale:
            return (self._out_channels, (H + 1) // 2, (W + 1) // 2)
        else:
            return (self._out_channels, H, W)
        

class DecoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        self.conv = nn.ConvTranspose2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=2,
            stride=2
        )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.blocks(self.conv(x))
        assert x.shape[1:] == self.get_output_shape()
        return x
    
    def get_output_shape(self):
        _, H, W = self._input_shape
        return (self._out_channels, H * 2, W * 2)
    

# Inputs an observation and outputs an action along with a latent representation of the observation
class Actor(nn.Module):
    def __init__(
            self,
            shape,
            num_actions,
            encoder_scale=1,
            encoder_channels=(16, 32, 32),
            encoder_num_res_blocks=1,
            dropout=0.0
    ):
        super().__init__()
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)
        self.final_encoder_shape = shape
        self.encoder = nn.Sequential(*conv_stack)
        
        self.actor_mean = nn.Sequential(
            nn.ReLU6(),
            nn.Linear(shape[0], num_actions)
        )
        self.num_actions = num_actions
        self.apply(weight_init)

    def forward(self, obs):
        out = self.encoder(obs).flatten(2).mean(-1)
        return self.actor_mean(out), out
    

# Takes a latent action as input and outputs the true action for it
class ActionDecoder(nn.Module):
    def __init__(self, latent_act_dim, true_act_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_act_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, true_act_dim)
        )
        
    def forward(self, latent_act):
        return self.model(latent_act)

class ObsActionDecoder(nn.Module):
    def __init__(self, latent_act_dim, obs_emb_dim, true_act_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_act_dim + obs_emb_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, true_act_dim)
        )
    
    def forward(self, latent_act, obs_emb):
        return self.model(
            torch.cat([latent_act, obs_emb], dim=-1)
        )
    
# Takes in observation and next observation and outputs the (latent) action taken in between    
class IDM(nn.Module):
    def __init__(
            self,
            shape,
            latent_act_dim,
            encoder_scale=1,
            encoder_channels=(16, 32, 64, 128, 256),
            encoder_num_res_blocks=1
    ):
        super().__init__()
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)
        
        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(in_features=math.prod(shape), out_features=latent_act_dim)
        )
        
    def forward(self, obs, next_obs):
        concat_obs = torch.cat([obs, next_obs], dim=1)
        latent_action = self.encoder(concat_obs)
        return latent_action
    
# Takes as input the current observation and the latent action and outputs the next observation
class FDM(nn.Module):
    def __init__(
            self,
            shape,
            latent_act_dim,
            encoder_scale=1,
            encoder_channels=(16, 32, 64, 128, 256),
            encoder_num_res_blocks=1
    ):
        super().__init__()
        self.initial_shape = shape

        # Encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(*conv_stack)
        self.final_encoder_shape = shape

        # Decoder
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels[::-1]:
            conv_seq = DecoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.decoder = nn.Sequential(
            *conv_stack,
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] * encoder_scale, self.initial_shape[0], kernel_size=1),
            nn.Tanh()
        )
        self.act_proj = nn.Linear(latent_act_dim, math.prod(self.final_encoder_shape))

    def forward(self, obs, latent_action):
        assert obs.ndim==4, "Expect shape [B, C, H, W]"
        obs_emb = self.encoder(obs)
        act_emb = self.act_proj(latent_action).reshape(-1 ,*self.final_encoder_shape)
        next_obs = self.decoder(
            torch.cat([obs_emb, act_emb], dim=1) # Concat across channel dimension
        )
        return next_obs
    
class LAPO(nn.Module):
    def __init__(
            self,
            shape,
            latent_act_dim,
            encoder_scale=1,
            encoder_channels=(16, 32, 64, 128, 256),
            encoder_num_res_blocks=1
    ):
        super().__init__()
        self.idm = IDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks
        )
        self.fdm = FDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks
        )
        self.latent_act_dim=latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        next_obs_pred = self.fdm(obs, latent_action)
        return next_obs_pred, latent_action
    
    @torch.no_grad()
    def label(self, obs, next_obs):
        return self.idm(obs, next_obs)
