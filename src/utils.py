import os
import random
import h5py
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from typing import List, Tuple, Dict, Iterator, Any
from shimmy import DmControlCompatibilityV0
from torch.utils.data import Dataset, IterableDataset
from .dcs import suite

def set_seed(seed, env=None, deterministic_torch=False) -> None:
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def get_optim_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    return [
        {"params": (p for p in model.parameters() if p.dim() < 2), "weight_decay": 0.0},
        {"params": (p for p in model.parameters() if p.dim() >= 2), "weight_decay": weight_decay}
    ] 

def get_grad_norm(model: nn.Module) -> float:
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    return torch.cat(grads).norm().item()

def soft_update(target: nn.Module, source: nn.Module, tau: float =1e-3) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

# Consider adding a stride
class DCSChunkedDataset(Dataset):
    def __init__(self, hdf5_path: str, frame_stack: int = 1) -> None:
        self.hdf5_path = hdf5_path
        self.frame_stack = frame_stack
        self.file = None 
        
        # Read metadata once
        with h5py.File(hdf5_path, "r") as f:
            self.traj_names = list(f.keys())
            first_traj = self.traj_names[0]
            self.traj_len = f[first_traj]["obs"].shape[0] # type: ignore
            self.img_hw = f.attrs["img_hw"]
            self.act_dim = f[first_traj]["actions"].shape[-1] # type: ignore
            self.state_dim = f[first_traj]["states"].shape[-1] # type: ignore

    def _lazy_init(self) -> None:
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return len(self.traj_names) * (self.traj_len - 1)

    def __get_padded_obs(self, traj_idx: int, idx: int) -> torch.Tensor:
        self._lazy_init()

        traj = self.traj_names[traj_idx]
        obs_ds = self.file[traj]["obs"] # type: ignore

        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1

        obs = obs_ds[min_obs_idx:max_obs_idx]  # numpy array [F, H, W, C] # type: ignore

        # Pad at beginning if fewer frames than frame_stack
        if obs.shape[0] < self.frame_stack: # type: ignore
            pad_frames = self.frame_stack - obs.shape[0] # type: ignore
            pad_obs = np.repeat(obs[0:1], pad_frames, axis=0) # type: ignore
            obs = np.concatenate([pad_obs, obs], axis=0) # type: ignore

        obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2) # [F, C, H, W]
        obs = obs.reshape(-1, *obs.shape[2:]) # [F*C, H, W] 
        return obs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_idx, transition_idx = divmod(idx, self.traj_len - 1)
        self._lazy_init()

        traj = self.traj_names[traj_idx]
        actions_ds = self.file[traj]["actions"] # type: ignore
        action = torch.tensor(actions_ds[transition_idx], dtype=torch.float32) # type: ignore

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)

        return obs, next_obs, action


# Consider adding a stride
class DCSChunkedHeatmapDataset(Dataset):
    def __init__(self, hdf5_path: str, frame_stack: int = 3, max_offset: int = 10) -> None:
       
        self.hdf5_path = hdf5_path
        self.frame_stack = frame_stack
        self.max_offset = max_offset
        self.file = None  

        # Read Meta-Data Only
        with h5py.File(hdf5_path, "r") as f:
            self.traj_names = list(f.keys())
            first_traj = self.traj_names[0]
            self.traj_len = f[first_traj]["obs"].shape[0] #type: ignore
            self.img_hw = f.attrs["img_hw"] 
            self.act_dim = f[first_traj]["actions"].shape[-1] # type: ignore 
            self.state_dim = f[first_traj]["states"].shape[-1] # type: ignore

        assert 1 <= max_offset < self.traj_len

    def _lazy_init(self) -> None:
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return len(self.traj_names) * (self.traj_len - self.max_offset)

    def __get_padded_obs(self, traj_idx: int, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._lazy_init()

        traj = self.traj_names[traj_idx]
        obs_ds = self.file[traj]["obs"] # type: ignore
        heatmap_ds = self.file[traj]["heatmaps"] # type: ignore

        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1

        obs = obs_ds[min_obs_idx:max_obs_idx]          # [F, H, W, C] # type: ignore
        heatmaps = heatmap_ds[min_obs_idx:max_obs_idx] # [F, H, W] # type: ignore

        # Pad at start if needed
        if obs.shape[0] < self.frame_stack: # type: ignore
            pad_frames = self.frame_stack - obs.shape[0] # type: ignore
            pad_obs = np.repeat(obs[0:1], pad_frames, axis=0) # type: ignore
            pad_heatmaps = np.repeat(heatmaps[0:1], pad_frames, axis=0) # type: ignore
            obs = np.concatenate([pad_obs, obs], axis=0) # type: ignore
            heatmaps = np.concatenate([pad_heatmaps, heatmaps], axis=0) # type: ignore

        obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).reshape(-1, *obs.shape[1:3]) # type: ignore
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        return obs, heatmaps

    def __getitem__(
            self,
            idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] :
        traj_idx, transition_idx = divmod(idx, self.traj_len - self.max_offset)

        self._lazy_init()

        traj = self.traj_names[traj_idx]
        actions_ds = self.file[traj]["actions"] # type: ignore

        action = torch.tensor(actions_ds[transition_idx], dtype=torch.float32) # type: ignore

        obs, _ = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs, next_hmaps = self.__get_padded_obs(traj_idx, transition_idx + 1)

        offset = random.randint(1, self.max_offset)
        future_obs, _ = self.__get_padded_obs(traj_idx, transition_idx + offset)

        return obs, next_obs, future_obs, next_hmaps, action, (offset - 1)


class DCSChunkedIterableDataset(IterableDataset):
    def __init__(self, hdf5_path: str, frame_stack: int = 3, max_offset: int = 10) -> None:
        super().__init__()
        self.hdf5_path = h5py.File(hdf5_path, 'r')
        self.frame_stack = frame_stack
        self.max_offset = max_offset

        self.traj_names = list(self.hdf5_path.keys())
        if not self.traj_names:
            raise ValueError("HDF5 file contains no trajectories.")

        # --- KEY IMPROVEMENT ---
        # Pre-scan all trajectory lengths. This is a fast metadata operation
        # and avoids loading the entire dataset into memory.
        self.traj_lengths = [self.hdf5_path[traj]["obs"].shape[0] for traj in self.traj_names]

        # Read other metadata
        first_traj = self.traj_names[0]
        self.img_hw = self.hdf5_path.attrs["img_hw"]
        self.act_dim = self.hdf5_path[first_traj]["actions"].shape[-1]
        self.state_dim = self.hdf5_path[first_traj]["states"].shape[-1]

    def _get_padded_obs(self, traj_name: str, idx: int) -> torch.Tensor:
        """Helper to get a frame-stacked observation, padding if necessary."""
        obs_ds = self.hdf5_path[traj_name]["obs"]

        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs_np = obs_ds[min_obs_idx:max_obs_idx]

        # Pad at the start if the window is incomplete
        if obs_np.shape[0] < self.frame_stack:
            pad_frames = self.frame_stack - obs_np.shape[0]
            pad_block = np.repeat(obs_np[0:1], pad_frames, axis=0)
            obs_np = np.concatenate([pad_block, obs_np], axis=0)

        # Convert to tensor, permute to [F, C, H, W], and flatten frames into channels
        obs_tensor = torch.from_numpy(obs_np).float().permute(0, 3, 1, 2)
        return obs_tensor.flatten(start_dim=0, end_dim=1)  # Shape: [F*C, H, W]

    def __iter__(self):
        """
        This iterator uses random sampling, mirroring your working example.
        It runs indefinitely, and the DataLoader handles workers and epochs.
        """
        while True:
            # 1. Randomly select a trajectory
            traj_idx = random.randrange(len(self.traj_names))
            traj_name = self.traj_names[traj_idx]
            current_traj_len = self.traj_lengths[traj_idx]

            # 2. Check if the trajectory is long enough to sample from
            if current_traj_len <= self.max_offset:
                continue  # Skip short trajectories and try again

            # 3. Randomly select a valid starting transition
            transition_idx = random.randint(0, current_traj_len - self.max_offset -1)

            # 4. Load only the data you need for this one sample
            offset = random.randint(1, self.max_offset)
            
            obs = self._get_padded_obs(traj_name, transition_idx)
            next_obs = self._get_padded_obs(traj_name, transition_idx + 1)
            future_obs = self._get_padded_obs(traj_name, transition_idx + offset)

            action_ds = self.hdf5_path[traj_name]["actions"]
            action = torch.from_numpy(action_ds[transition_idx]).float()
            
            # Note: Removed states and heatmaps as per your previous requests.
            # You can easily add them back here if needed.
            
            yield obs, next_obs, future_obs, action, (offset - 1)

    def __del__(self):
        """Ensures the file handle is closed when the object is garbage collected."""
        self.hdf5_path.close()


class DCSChunkedLAOMDataset(Dataset):
    def __init__(self, hdf5_path: str, frame_stack: int = 3, max_offset: int = 10) -> None:
       
        self.hdf5_path = hdf5_path
        self.frame_stack = frame_stack
        self.max_offset = max_offset
        self.file = None  

        # Read Meta-Data Only
        with h5py.File(hdf5_path, "r") as f:
            self.traj_names = list(f.keys())
            first_traj = self.traj_names[0]
            self.traj_len = f[first_traj]["obs"].shape[0] #type: ignore
            self.img_hw = f.attrs["img_hw"] 
            self.act_dim = f[first_traj]["actions"].shape[-1] # type: ignore 
            self.state_dim = f[first_traj]["states"].shape[-1] # type: ignore

        assert 1 <= max_offset < self.traj_len

    def _lazy_init(self) -> None:
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return len(self.traj_names) * (self.traj_len - self.max_offset)

    def __get_padded_obs(self, traj_idx: int, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._lazy_init()

        traj = self.traj_names[traj_idx]
        obs_ds = self.file[traj]["obs"] # type: ignore
        heatmap_ds = self.file[traj]["heatmaps"] # type: ignore

        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1

        obs = obs_ds[min_obs_idx:max_obs_idx]          # [F, H, W, C] # type: ignore
        heatmaps = heatmap_ds[min_obs_idx:max_obs_idx] # [F, H, W] # type: ignore

        # Pad at start if needed
        if obs.shape[0] < self.frame_stack: # type: ignore
            pad_frames = self.frame_stack - obs.shape[0] # type: ignore
            pad_obs = np.repeat(obs[0:1], pad_frames, axis=0) # type: ignore
            pad_heatmaps = np.repeat(heatmaps[0:1], pad_frames, axis=0) # type: ignore
            obs = np.concatenate([pad_obs, obs], axis=0) # type: ignore
            heatmaps = np.concatenate([pad_heatmaps, heatmaps], axis=0) # type: ignore

        obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).reshape(-1, *obs.shape[1:3]) # type: ignore
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        return obs, heatmaps

    def __getitem__(
            self,
            idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] :
        traj_idx, transition_idx = divmod(idx, self.traj_len - self.max_offset)

        self._lazy_init()

        traj = self.traj_names[traj_idx]
        actions_ds = self.file[traj]["actions"] # type: ignore
        states_ds = self.file[traj]["states"] # type: ignore

        action = torch.tensor(actions_ds[transition_idx], dtype=torch.float32) # type: ignore
        state = torch.tensor(states_ds[transition_idx], dtype=torch.float32) # type: ignore

        obs, _ = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs, _ = self.__get_padded_obs(traj_idx, transition_idx + 1)

        offset = random.randint(1, self.max_offset)
        future_obs, _ = self.__get_padded_obs(traj_idx, transition_idx + offset)

        return obs, next_obs, future_obs, state, action, (offset - 1)
    

def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset # type: ignore
    dataset.file = None # type: ignore


def normalise_img(img: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    return ((img / 255.0) - 0.5) * 2.0

def unnormalise_img(img):
    return ((img / 2.0) + 0.5) * 255.0

def weight_init(m: nn.Parameter | nn.Module) -> None:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None and hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    

class SelectPixelsObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = self.env.observation_space["pixels"] # type: ignore

    def observation(self, obs: Dict[str, Any]) -> np.ndarray: # type: ignore
        return obs["pixels"]
    
class FlattenStackedFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape # [F, H, W, C]
        new_shape = old_shape[1:-1] + (old_shape[0] * old_shape[-1],) # old_shape: [F, H, W, 3] #type: ignore
        # new_shape = (old_shape[0] * old_shape[-1],) + old_shape[1:-1] 
        # new_shape=(64, 64, 9)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray: # type: ignore
        obs = obs.transpose((1, 2, 0, 3)) # H, W, F, C
        obs = obs.reshape(*obs.shape[:2], -1) # H, W, F*C
        return obs
    
def create_env_from_df(
        hdf5_path: str,
        backgrounds_path: str,
        backgrounds_split: str,
        frame_stack: int = 1,
        pixels_only: bool = True,
        flatten_frames: bool = True,
        difficulty: Any = None,
    ) -> gym.Env:
    with h5py.File(hdf5_path, "r") as df:
        dm_env = suite.load(
            domain_name=df.attrs["domain_name"],
            task_name=df.attrs["task_name"],
            difficulty=df.attrs["difficulty"] if difficulty is None else difficulty,
            dynamic=df.attrs["dynamic"], # type: ignore
            background_dataset_path=backgrounds_path,
            background_dataset_videos=backgrounds_split,
            pixels_only=pixels_only,
            render_kwargs=dict(height=df.attrs["img_hw"], width=df.attrs["img_hw"])
        )
        env = DmControlCompatibilityV0(dm_env)
        env = gym.wrappers.ClipAction(env)

        if pixels_only:
            env = SelectPixelsObsWrapper(env)

        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            if flatten_frames:
                env = FlattenStackedFrames(env)
    return env


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
    B, _, H, W = pred.size()
    pred, target = pred.reshape(shape=(B, -1, 3, H, W)), target.reshape(shape=(B, -1, 3, H, W))
    heatmap = heatmap.unsqueeze(dim=2)
    heatmap = torch.where(heatmap > 0.5, 1.0, 1e-2) # experimental
    squared_error = (target - pred) ** 2
    return (heatmap * squared_error).mean()