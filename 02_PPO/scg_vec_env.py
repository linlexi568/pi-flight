from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# CRITICAL: Isaac Gym 必须在 torch 之前导入！
# stable_baselines3 会导入 torch，所以我们必须先导入 Isaac Gym
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PI_FLIGHT_DIR = ROOT / "01_pi_flight"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PI_FLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(PI_FLIGHT_DIR))

# 先导入 Isaac Gym 相关模块（内部会正确处理 torch 导入顺序）
from envs.isaac_gym_drone_env import IsaacGymDroneEnv  # type: ignore  # noqa: E402
from utils.reward_scg_exact import SCGExactRewardCalculator  # type: ignore  # noqa: E402
from utilities.trajectory_presets import get_scg_trajectory_config

# 现在可以安全导入 torch 和 stable_baselines3
import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


@dataclass(frozen=True)
class TrajectorySpec:
    task: str
    params: Dict[str, float]
    initial_xyz: Sequence[float]


def default_trajectory_params() -> Dict[str, Dict[str, float]]:
    catalog: Dict[str, Dict[str, float]] = {}
    for task in ("hover", "figure8", "circle", "helix", "square"):
        cfg = get_scg_trajectory_config(task)
        params = dict(cfg.params)
        params["center"] = cfg.center
        catalog[task] = params
    return catalog


class IsaacSCGVecEnv(VecEnv):
    """Stable-Baselines VecEnv wrapper with SCG reward."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str,
        task: str,
        duration: float,
        traj_params: Optional[Dict[str, float]] = None,
        initial_xyz: Optional[Iterable[float]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.task = task
        self.duration = float(duration)
        cfg = get_scg_trajectory_config(task, overrides=traj_params or {})
        self.traj_params = dict(cfg.params)
        init_xyz = initial_xyz or cfg.center
        self.initial_xyz = torch.as_tensor(init_xyz, device=self.device, dtype=torch.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        super().__init__(self.num_envs, self.observation_space, self.action_space)

        self.env_pool = IsaacGymDroneEnv(
            num_envs=self.num_envs,
            device=device,
            headless=True,
            duration_sec=self.duration,
        )
        self.control_freq = float(getattr(self.env_pool, "control_freq", 48.0))
        self.dt = 1.0 / self.control_freq
        self.max_steps = int(self.duration * self.control_freq)

        self.reward_calc = SCGExactRewardCalculator(num_envs=self.num_envs, device=device)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.env_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.action_scale = torch.tensor([20.0, 10.0, 10.0, 10.0], device=self.device)  # [u_fz, u_tx, u_ty, u_tz]
        self.actions_buf: Optional[torch.Tensor] = None

        self.env_pool.reset()

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> np.ndarray:  # type: ignore[override]
        self.env_pool.reset()
        self.reward_calc.reset(self.num_envs)
        self.env_steps.zero_()
        self.episode_returns.zero_()
        obs_dict = self.env_pool.get_obs()
        obs_tensor = self._format_obs(obs_dict, self.env_steps.float() * self.dt)
        return obs_tensor.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions_buf = torch.as_tensor(actions, device=self.device, dtype=torch.float32)

    def step_wait(self):  # type: ignore[override]
        if self.actions_buf is None:
            raise RuntimeError("step_wait called before step_async")

        scaled_actions = self.actions_buf * self.action_scale
        zeros = torch.zeros(self.num_envs, device=self.device)
        forces_6d = torch.stack(
            [
                zeros,
                zeros,
                scaled_actions[:, 0],
                scaled_actions[:, 1],
                scaled_actions[:, 2],
                scaled_actions[:, 3],
            ],
            dim=1,
        )

        obs_dict, _, done_flags, _ = self.env_pool.step(forces_6d)
        dones = torch.as_tensor(done_flags, device=self.device, dtype=torch.bool)

        pos = torch.as_tensor(obs_dict["position"], device=self.device)
        vel = torch.as_tensor(obs_dict["velocity"], device=self.device)
        quat = torch.as_tensor(obs_dict["orientation"], device=self.device)
        omega = torch.as_tensor(obs_dict["angular_velocity"], device=self.device)

        t_tensor = self.env_steps.float() * self.dt
        targets = self._compute_targets(t_tensor)

        rewards = self.reward_calc.compute_step(
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            target_pos=targets,
            action=scaled_actions,
            done_mask=dones,
        )

        self.env_steps += 1
        self.episode_returns += rewards

        episode_returns_snapshot = self.episode_returns.clone()
        episode_lengths_snapshot = self.env_steps.clone()

        if dones.any():
            reset_ids = torch.nonzero(dones).squeeze(-1)
            self.env_steps[reset_ids] = 0
            self.episode_returns[reset_ids] = 0.0

        obs_next = obs_dict
        if dones.any():
            obs_next = self.env_pool.get_obs()

        obs_tensor = self._format_obs(obs_next, self.env_steps.float() * self.dt)

        infos = [{} for _ in range(self.num_envs)]
        if dones.any():
            done_indices = torch.nonzero(dones).squeeze(-1).cpu().numpy()
            terminal_obs_tensor = self._format_obs(obs_dict, t_tensor)
            for idx in done_indices:
                infos[idx]["terminal_observation"] = terminal_obs_tensor[idx].cpu().numpy()
                infos[idx]["episode"] = {
                    "r": float(episode_returns_snapshot[idx].item()),
                    "l": int(episode_lengths_snapshot[idx].item()),
                }

        return (
            obs_tensor.cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            infos,
        )

    def close(self) -> None:  # type: ignore[override]
        if hasattr(self, "env_pool"):
            self.env_pool.close()

    # ------------------------------------------------------------------
    # VecEnv protocol helpers
    # ------------------------------------------------------------------
    def env_is_wrapped(self, wrapper_class, indices=None):  # type: ignore[override]
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):  # type: ignore[override]
        return [None] * (self.num_envs if indices is None else len(indices))

    def get_attr(self, attr_name, indices=None):  # type: ignore[override]
        value = getattr(self, attr_name, None)
        n = self.num_envs if indices is None else len(indices)
        return [value] * n

    def set_attr(self, attr_name, value, indices=None):  # type: ignore[override]
        setattr(self, attr_name, value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_targets(self, t_tensor: torch.Tensor) -> torch.Tensor:
        base = self.initial_xyz.unsqueeze(0).expand(self.num_envs, -1)

        if self.task == "hover":
            return base

        plane = str(self.traj_params.get("plane", "xy")).lower()
        idx_a, idx_b = self._plane_indices(plane)
        delta = torch.zeros(self.num_envs, 3, device=self.device)

        if self.task == "figure8":
            scale = float(self.traj_params.get("scale", 0.8))
            A = float(self.traj_params.get("A", scale))
            B = float(self.traj_params.get("B", scale))
            period = float(self.traj_params.get("period", 5.0))
            w = 2.0 * np.pi / period
            x = A * torch.sin(w * t_tensor)
            y = B * torch.sin(w * t_tensor) * torch.cos(w * t_tensor)
            delta[:, idx_a] = x
            delta[:, idx_b] = y
            return base + delta

        if self.task == "circle":
            R = float(self.traj_params.get("R", 0.9))
            period = float(self.traj_params.get("period", 5.0))
            w = 2.0 * np.pi / period
            x = R * torch.cos(w * t_tensor)
            y = R * torch.sin(w * t_tensor)
            delta[:, idx_a] = x
            delta[:, idx_b] = y
            return base + delta

        if self.task == "helix":
            R = float(self.traj_params.get("R", 0.7))
            period = float(self.traj_params.get("period", 10.0))
            vz = float(self.traj_params.get("v_z", 0.1))
            w = 2.0 * np.pi / period
            x = R * torch.cos(w * t_tensor)
            y = R * torch.sin(w * t_tensor)
            z = vz * t_tensor
            delta[:, idx_a] = x
            delta[:, idx_b] = y
            delta[:, 2] += z
            return base + delta

        if self.task == "square":
            scale = float(self.traj_params.get("scale", self.traj_params.get("side", 0.8)))
            period = float(self.traj_params.get("period", 8.0))
            segment_period = period / 4.0
            traverse_speed = scale / max(segment_period, 1e-6)

            t_mod = torch.remainder(t_tensor, period)
            seg_float = torch.clamp(torch.floor(t_mod / segment_period), max=3)
            seg_time = t_mod - seg_float * segment_period
            seg_pos = traverse_speed * seg_time

            coord_a = torch.zeros_like(t_tensor)
            coord_b = torch.zeros_like(t_tensor)

            mask0 = seg_float == 0  # (0,0) -> (0, scale)
            coord_a[mask0] = 0.0
            coord_b[mask0] = seg_pos[mask0]

            mask1 = seg_float == 1  # (0, scale) -> (-scale, scale)
            coord_a[mask1] = -seg_pos[mask1]
            coord_b[mask1] = scale

            mask2 = seg_float == 2  # (-scale, scale) -> (-scale, 0)
            coord_a[mask2] = -scale
            coord_b[mask2] = scale - seg_pos[mask2]

            mask3 = seg_float == 3  # (-scale, 0) -> (0, 0)
            coord_a[mask3] = -scale + seg_pos[mask3]
            coord_b[mask3] = 0.0

            delta[:, idx_a] = coord_a
            delta[:, idx_b] = coord_b
            return base + delta

        return base

    @staticmethod
    def _plane_indices(plane: str) -> Tuple[int, int]:
        axis = {"x": 0, "y": 1, "z": 2}
        plane = (plane or "xy").lower()
        if len(plane) != 2 or plane[0] == plane[1]:
            return 0, 1
        return axis.get(plane[0], 0), axis.get(plane[1], 1)

    def _format_obs(self, obs_dict: Dict[str, np.ndarray], t_tensor: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(obs_dict["position"], device=self.device)
        vel = torch.as_tensor(obs_dict["velocity"], device=self.device)
        quat = torch.as_tensor(obs_dict["orientation"], device=self.device)
        omega = torch.as_tensor(obs_dict["angular_velocity"], device=self.device)

        targets = self._compute_targets(t_tensor)
        pos_err = pos - targets
        rpy = self._quat_to_rpy(quat)
        return torch.cat([pos_err, vel, omega, rpy], dim=1)

    def _quat_to_rpy(self, quat: torch.Tensor) -> torch.Tensor:
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        sinp_clamped = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp_clamped)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=1)


__all__ = ["IsaacSCGVecEnv", "TrajectorySpec", "default_trajectory_params"]
