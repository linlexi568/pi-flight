#!/usr/bin/env python3
"""
PPO Baseline for Drone Control
使用Stable-Baselines3实现,作为黑盒NN baseline对比
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / '01_pi_flight'))

# Isaac Gym 必须在 torch 之前导入
_GYM_PATH = _ROOT / 'isaacgym' / 'python'
if _GYM_PATH.exists():
    sys.path.insert(0, str(_GYM_PATH))
    try:
        from isaacgym import gymapi
    except ImportError:
        print("[WARNING] Isaac Gym 未安装，PPO 训练可能无法运行")

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# 导入 π-Flight 的组件
try:
    from envs.isaac_gym_drone_env import IsaacGymDroneEnv
    from utils.reward_stepwise import StepwiseRewardCalculator
    from utilities.reward_profiles import get_reward_profile
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    raise

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class _FixedConfig:
    """
    Hardcoded configuration for PPO baseline.
    Edit this class to change parameters.
    NO CLI ARGUMENTS ALLOWED.
    """
    # --- Experiment ---
    mode = 'train'  # 'train' or 'test'
    experiment_name = 'ppo_baseline_v1'
    seed = 42
    
    # --- Environment ---
    task = 'figure8'  # 'hover', 'figure8', 'circle', 'helix'
    duration = 10.0   # seconds per episode
    isaac_num_envs = 512  # Number of parallel environments in Isaac Gym
    
    # --- Reward ---
    # Options: 'safety_first', 'tracking_first', 'balanced', 'robustness_stability'
    reward_profile = 'balanced'
    
    # --- Training ---
    total_timesteps = 1_000_000
    learning_rate = 3e-4
    n_steps = 2048        # Steps per update per env
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.0
    
    # --- System ---
    device = 'cuda:0'  # 'cpu' or 'cuda:0'
    
    # --- Trajectory Params (matches BatchEvaluator defaults) ---
    traj_params = {
        'figure8': {'A': 0.8, 'B': 0.5, 'period': 12.0},
        'circle': {'R': 0.9, 'period': 10.0},
        'helix': {'R': 0.7, 'period': 10.0, 'v_z': 0.15},
        'hover': {}
    }

    @property
    def trajectory_config(self):
        return {
            'type': self.task,
            'params': self.traj_params.get(self.task, {}),
            'initial_xyz': [0.0, 0.0, 1.0]
        }

# -----------------------------------------------------------------------------
# Environment Wrapper
# -----------------------------------------------------------------------------
class DroneControlEnv(VecEnv):
    """
    VecEnv wrapper for IsaacGymDroneEnv to be compatible with Stable-Baselines3.
    Manages 512 parallel environments directly.
    """
    def __init__(self, cfg: _FixedConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_envs = cfg.isaac_num_envs
        
        # Define spaces (Gymnasium)
        # Action: [u_fz, u_tx, u_ty, u_tz] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Observation: [pos_err(3), vel(3), ang_vel(3), rpy_err(3)] -> 12 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        # Initialize VecEnv
        super().__init__(self.num_envs, self.observation_space, self.action_space)
        
        # Initialize Isaac Gym Environment
        print(f"[PPO Env] Initializing IsaacGymDroneEnv with {self.num_envs} envs...")
        self.env_pool = IsaacGymDroneEnv(
            num_envs=self.num_envs,
            device=cfg.device,
            headless=True,
            duration_sec=cfg.duration
        )
        
        # Initialize Reward Calculator
        weights, ks = get_reward_profile(cfg.reward_profile)
        # Estimate dt from env control freq (usually 48Hz)
        try:
            self.control_freq = self.env_pool.control_freq
        except:
            self.control_freq = 48.0
        self.dt = 1.0 / self.control_freq
        
        self.reward_calc = StepwiseRewardCalculator(
            weights, ks, dt=self.dt, num_envs=self.num_envs, device=cfg.device
        )
        
        # State tracking
        self.env_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.actions_buf = None
        
        # Initial reset
        self.env_pool.reset()
        self.reward_calc.reset_components()
        
    def step_async(self, actions):
        # Store actions for step_wait
        # Actions from SB3 are numpy arrays, convert to tensor
        self.actions_buf = torch.as_tensor(actions, device=self.device, dtype=torch.float32)

    def step_wait(self):
        # 1. Scale actions
        # PPO output [-1, 1] -> Physical units
        # u_fz * 20.0 -> [-20, 20] N
        # u_tau * 10.0 -> [-10, 10] Nm
        
        u_fz = self.actions_buf[:, 0] * 20.0
        u_tx = self.actions_buf[:, 1] * 10.0
        u_ty = self.actions_buf[:, 2] * 10.0
        u_tz = self.actions_buf[:, 3] * 10.0
        
        # 2. Prepare forces for IsaacGymDroneEnv [0, 0, fz, tx, ty, tz]
        # Note: IsaacGymDroneEnv expects [N, 6]
        zeros = torch.zeros((self.num_envs, 2), device=self.device)
        forces_6d = torch.stack([
            torch.zeros(self.num_envs, device=self.device), # fx
            torch.zeros(self.num_envs, device=self.device), # fy
            u_fz,
            u_tx,
            u_ty,
            u_tz
        ], dim=1)
        
        # 3. Step Physics
        # obs_terminal is the state AFTER physics step, but BEFORE reset (if done)
        obs_terminal_dict, _, dones, _ = self.env_pool.step(forces_6d)
        
        # 4. Calculate Rewards
        # Calculate targets for each env based on its current step count
        t = self.env_steps.float() * self.dt
        targets = self._compute_targets(t) # [N, 3]
        
        # Extract state tensors
        pos = torch.as_tensor(obs_terminal_dict['position'], device=self.device)
        vel = torch.as_tensor(obs_terminal_dict['velocity'], device=self.device)
        quat = torch.as_tensor(obs_terminal_dict['orientation'], device=self.device)
        omega = torch.as_tensor(obs_terminal_dict['angular_velocity'], device=self.device)
        
        # Compute reward
        rewards = self.reward_calc.compute_step(
            pos=pos,
            target=targets,
            vel=vel,
            omega=omega,
            actions=forces_6d, # Pass 6D forces
            done_mask=dones # Dones from this step
        )
        
        # 5. Handle Resets and Observations
        # Update steps
        self.env_steps += 1
        
        # If done, reset steps for those envs
        if dones.any():
            reset_ids = torch.nonzero(dones).squeeze(-1)
            self.env_steps[reset_ids] = 0
            
            # Get current (reset) observations
            obs_reset_dict = self.env_pool.get_obs()
            final_obs_dict = obs_reset_dict
        else:
            final_obs_dict = obs_terminal_dict
            
        # 6. Format Observations for PPO [N, 12]
        final_obs_tensor = self._format_obs(final_obs_dict, t if not dones.any() else self.env_steps.float() * self.dt)
        
        # 7. Construct Infos (Terminal observations)
        infos = [{} for _ in range(self.num_envs)]
        if dones.any():
            # Format terminal obs
            term_obs_tensor = self._format_obs(obs_terminal_dict, t)
            
            done_indices = torch.nonzero(dones).squeeze(-1).cpu().numpy()
            for idx in done_indices:
                infos[idx]['terminal_observation'] = term_obs_tensor[idx].cpu().numpy()
                infos[idx]['TimeLimit.truncated'] = (self.env_steps[idx] >= (self.cfg.duration / self.dt)) # Approx check
        
        return final_obs_tensor.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), infos

    def _compute_targets(self, t_tensor: torch.Tensor) -> torch.Tensor:
        # Vectorized target computation
        # t_tensor: [N]
        traj = self.cfg.trajectory_config
        tp = traj['type']
        init = torch.tensor(traj['initial_xyz'], device=self.device)
        params = traj['params']
        
        if tp == 'hover':
            return init.expand(self.num_envs, 3)
        elif tp == 'figure8':
            A = float(params.get('A', 0.8))
            B = float(params.get('B', 0.5))
            period = float(params.get('period', 12.0))
            w = 2.0 * np.pi / period
            
            x = A * torch.sin(w * t_tensor)
            y = B * torch.sin(w * t_tensor) * torch.cos(w * t_tensor)
            z = torch.zeros_like(t_tensor)
            
            delta = torch.stack([x, y, z], dim=1)
            return init + delta
        # Add other types if needed
        return init.expand(self.num_envs, 3)

    def _format_obs(self, obs_dict, t_tensor):
        # Convert dict obs to [N, 12] vector
        # [pos_err(3), vel(3), ang_vel(3), rpy_err(3)]
        
        pos = torch.as_tensor(obs_dict['position'], device=self.device)
        vel = torch.as_tensor(obs_dict['velocity'], device=self.device)
        quat = torch.as_tensor(obs_dict['orientation'], device=self.device)
        omega = torch.as_tensor(obs_dict['angular_velocity'], device=self.device)
        
        # Target
        target_pos = self._compute_targets(t_tensor)
        pos_err = pos - target_pos
        
        # RPY Error (Simplified: assume target RPY is 0 for now, or just feed RPY)
        rpy = self._quat_to_rpy(quat)
        
        # Obs: [pos_err, vel, omega, rpy]
        obs = torch.cat([pos_err, vel, omega, rpy], dim=1)
        return obs

    def _quat_to_rpy(self, q):
        # q: [N, 4] (x, y, z, w)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (np.pi / 2), torch.asin(sinp))
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=1)

    def reset(self):
        self.env_pool.reset()
        self.env_steps.zero_()
        self.reward_calc.reset_components()
        
        obs_dict = self.env_pool.get_obs()
        t = torch.zeros(self.num_envs, device=self.device)
        return self._format_obs(obs_dict, t).cpu().numpy()

    def close(self):
        self.env_pool.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            n = self.num_envs
        else:
            n = len(indices)
        return [False] * n

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            if indices is None:
                n = self.num_envs
            else:
                n = len(indices)
            return [val] * n
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)


def train_ppo_baseline(
    trajectory='figure8',
    duration=10,
    isaac_num_envs=512,
    reward_profile='balanced',
    total_timesteps=1_000_000,
    n_envs=1, # Ignored, using isaac_num_envs internally
    save_dir='./02_PPO/results',
    eval_freq=10000,
):
    """
    训练PPO baseline
    """
    print(f"[PPO Baseline] 开始训练")
    print(f"  轨迹: {trajectory}, 时长: {duration}s")
    print(f"  奖励 profile: {reward_profile}")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  Isaac 并行环境: {isaac_num_envs}")
    
    # 创建保存目录
    save_path = Path(save_dir) / f"{trajectory}_{reward_profile}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 创建配置对象
    cfg = _FixedConfig()
    cfg.task = trajectory
    cfg.duration = duration
    cfg.isaac_num_envs = isaac_num_envs
    cfg.reward_profile = reward_profile
    
    # 创建环境 (VecEnv)
    env = DroneControlEnv(cfg)
    
    # 定义PPO模型
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,          # 每次更新收集的步数
        batch_size=4096,       # Batch size for PPO update (larger for Isaac)
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(save_path / 'tensorboard'),
        device='cuda',
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // isaac_num_envs,
        save_path=str(save_path / 'checkpoints'),
        name_prefix='ppo_model',
    )
    
    # 训练!
    print("[PPO Baseline] 开始训练...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True,
    )
    
    # 保存最终模型
    final_model_path = save_path / 'final_model'
    model.save(str(final_model_path))
    print(f"[PPO Baseline] 训练完成! 模型已保存: {final_model_path}")
    
    # 清理
    env.close()
    
    return model, save_path


def evaluate_ppo_baseline(
    model_path,
    trajectory='figure8',
    duration=10,
    isaac_num_envs=512,
    reward_profile='balanced',
    n_eval=100
):
    """
    评估训练好的PPO模型
    """
    print(f"[PPO Baseline] 评估模型: {model_path}")
    print(f"  轨迹: {trajectory}, profile: {reward_profile}")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建配置
    cfg = _FixedConfig()
    cfg.task = trajectory
    cfg.duration = duration
    cfg.isaac_num_envs = isaac_num_envs
    cfg.reward_profile = reward_profile
    
    # 创建评估环境
    env = DroneControlEnv(cfg)
    
    # 运行评估
    # 由于是 VecEnv，我们一次评估 isaac_num_envs 个 episode
    # 我们运行 n_eval // isaac_num_envs 次 batch
    
    n_batches = max(1, n_eval // isaac_num_envs)
    all_rewards = []
    all_lengths = []
    
    for i in range(n_batches):
        obs = env.reset()
        # VecEnv reset returns obs
        
        # Track rewards per env
        current_rewards = np.zeros(isaac_num_envs)
        current_lengths = np.zeros(isaac_num_envs)
        dones = np.zeros(isaac_num_envs, dtype=bool)
        
        # Run until all done? No, Isaac runs continuously.
        # We run for fixed duration or until done.
        # Let's run for duration steps.
        max_steps = int(duration / env.dt)
        
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, step_dones, infos = env.step(action)
            
            # Accumulate rewards for non-done envs
            # If done, the reward returned is for the last step.
            # We should add it.
            
            # Note: if an env is already done in this batch, we shouldn't count it again?
            # But Isaac auto-resets.
            # Let's just collect first episode from each env.
            
            active = ~dones
            current_rewards[active] += rewards[active]
            current_lengths[active] += 1
            
            # Update dones
            new_dones = step_dones & active
            dones = dones | step_dones
            
            if dones.all():
                break
        
        all_rewards.extend(current_rewards)
        all_lengths.extend(current_lengths)
        
        print(f"  Batch {i+1}/{n_batches}: Mean Reward={np.mean(current_rewards):.3f}")
    
    env.close()
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_length = np.mean(all_lengths)
    
    print(f"\n[PPO Baseline] 评估结果 ({len(all_rewards)} episodes):")
    print(f"  平均奖励: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  平均长度: {mean_length:.1f}")
    
    return mean_reward, std_reward


def main():
    """主入口 - 使用固定配置运行"""
    cfg = _FixedConfig
    
    print("=" * 80)
    print("PPO Baseline 训练/评估")
    print("=" * 80)
    print(f"模式: {cfg.mode}")
    print(f"任务: {cfg.task} (轨迹={cfg.task}, 时长={cfg.duration}s)")
    print(f"奖励 profile: {cfg.reward_profile}")
    print("=" * 80)
    print()
    
    if cfg.mode == 'train':
        train_ppo_baseline(
            trajectory=cfg.task,
            duration=cfg.duration,
            isaac_num_envs=cfg.isaac_num_envs,
            reward_profile=cfg.reward_profile,
            total_timesteps=cfg.total_timesteps,
            n_envs=1,
            save_dir='./02_PPO/results',
            eval_freq=10000,
        )
    
    elif cfg.mode == 'eval':
        # Find model path
        model_path = f"./02_PPO/results/{cfg.task}_{cfg.reward_profile}/final_model"
        if not os.path.exists(model_path + ".zip"):
             print(f"Model not found at {model_path}")
             return

        evaluate_ppo_baseline(
            model_path=model_path,
            trajectory=cfg.task,
            duration=cfg.duration,
            isaac_num_envs=cfg.isaac_num_envs,
            reward_profile=cfg.reward_profile,
            n_eval=100,
        )
    
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == '__main__':
    main()

