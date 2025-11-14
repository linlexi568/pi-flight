#!/usr/bin/env python3
"""
PPO Baseline for Drone Control
使用Stable-Baselines3实现,作为黑盒NN baseline对比
"""

import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import argparse
from pathlib import Path

# 导入你的环境评估器
import sys
sys.path.append(str(Path(__file__).parent))
from batch_evaluation import BatchEvaluator


class DroneControlEnv(gym.Env):
    """
    将你的Isaac Gym环境包装成Gym接口
    """
    def __init__(self, task='circle', render=False):
        super().__init__()
        
        self.task = task
        
        # 观察空间: 根据你的DSL变量定义
        # 假设使用12维状态 (pos_err_xyz, vel_xyz, ang_vel_xyz, rpy_err)
        obs_dim = 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 动作空间: 直接输出控制量 u_tx, u_ty, u_tz
        # 归一化到[-1, 1]
        action_dim = 3
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # 初始化评估器 (单环境版本)
        self.evaluator = BatchEvaluator(
            num_envs=1,  # PPO训练时每个worker只用1个环境
            device='cuda',
            task=task
        )
        
        self.max_steps = 300  # 每个episode最大步数
        self.current_step = 0
        self.state = None
        
    def reset(self):
        """重置环境"""
        self.current_step = 0
        # 从评估器获取初始状态
        self.state = self.evaluator.reset()
        return self._get_obs()
    
    def _get_obs(self):
        """提取观察值 (12维状态向量)"""
        # 根据你的环境实现,提取相关状态
        # 示例: pos_err, vel, ang_vel, rpy_err
        obs = np.array([
            self.state['pos_err_x'],
            self.state['pos_err_y'],
            self.state['pos_err_z'],
            self.state['vel_x'],
            self.state['vel_y'],
            self.state['vel_z'],
            self.state['ang_vel_x'],
            self.state['ang_vel_y'],
            self.state['ang_vel_z'],
            self.state['err_p_roll'],
            self.state['err_p_pitch'],
            self.state['err_p_yaw'],
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        # 将归一化的动作[-1,1]映射到实际控制量
        # 假设控制量范围是[-10, 10]
        action_scaled = action * 10.0
        
        # 应用动作,获取下一个状态和奖励
        next_state, reward, info = self.evaluator.step(action_scaled)
        self.state = next_state
        
        # 判断是否结束
        done = (self.current_step >= self.max_steps) or info.get('crashed', False)
        
        return self._get_obs(), reward, done, info
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'evaluator'):
            del self.evaluator


def make_env(task='circle', rank=0):
    """创建环境的工厂函数 (用于并行)"""
    def _init():
        env = DroneControlEnv(task=task)
        env = Monitor(env)  # 记录统计信息
        return env
    return _init


def train_ppo_baseline(
    task='circle',
    total_timesteps=1_000_000,
    n_envs=8,
    save_dir='./ppo_baseline',
    eval_freq=10000,
):
    """
    训练PPO baseline
    
    Args:
        task: 任务类型 ('circle', 'figure8', 'hover_wind')
        total_timesteps: 总训练步数
        n_envs: 并行环境数
        save_dir: 模型保存目录
        eval_freq: 评估频率
    """
    print(f"[PPO Baseline] 开始训练 - 任务: {task}")
    print(f"[PPO Baseline] 总步数: {total_timesteps:,}, 并行环境: {n_envs}")
    
    # 创建保存目录
    save_path = Path(save_dir) / task
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 创建并行环境
    if n_envs > 1:
        env = SubprocVecEnv([make_env(task, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(task)])
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(task)])
    
    # 定义PPO模型
    # 使用标准超参数 (可以根据需要调整)
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,          # 每次更新收集的步数
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # 熵系数 (鼓励探索)
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(save_path / 'tensorboard'),
        device='cuda',
    )
    
    # 设置回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / 'best_model'),
        log_path=str(save_path / 'eval_logs'),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(save_path / 'checkpoints'),
        name_prefix='ppo_model',
    )
    
    # 训练!
    print("[PPO Baseline] 开始训练...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    # 保存最终模型
    final_model_path = save_path / 'final_model'
    model.save(str(final_model_path))
    print(f"[PPO Baseline] 训练完成! 模型已保存: {final_model_path}")
    
    # 清理
    env.close()
    eval_env.close()
    
    return model, save_path


def evaluate_ppo_baseline(model_path, task='circle', n_eval=100):
    """
    评估训练好的PPO模型
    
    Args:
        model_path: 模型路径
        task: 任务类型
        n_eval: 评估episode数
    
    Returns:
        平均奖励, 标准差
    """
    print(f"[PPO Baseline] 评估模型: {model_path}")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建评估环境
    env = DummyVecEnv([make_env(task)])
    
    # 运行评估
    episode_rewards = []
    episode_lengths = []
    
    for i in range(n_eval):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (i + 1) % 10 == 0:
            print(f"  Episode {i+1}/{n_eval}: Reward={episode_reward:.3f}")
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n[PPO Baseline] 评估结果 ({n_eval} episodes):")
    print(f"  平均奖励: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  平均长度: {mean_length:.1f}")
    
    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description='PPO Baseline for Drone Control')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='训练或评估模式')
    parser.add_argument('--task', type=str, default='circle',
                        choices=['circle', 'figure8', 'hover_wind'],
                        help='任务类型')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='总训练步数')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='并行环境数')
    parser.add_argument('--save-dir', type=str, default='./ppo_baseline',
                        help='模型保存目录')
    parser.add_argument('--model-path', type=str, default=None,
                        help='评估时的模型路径')
    parser.add_argument('--n-eval', type=int, default=100,
                        help='评估episode数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ppo_baseline(
            task=args.task,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_dir=args.save_dir,
        )
    
    elif args.mode == 'eval':
        if args.model_path is None:
            args.model_path = f"{args.save_dir}/{args.task}/final_model"
        
        evaluate_ppo_baseline(
            model_path=args.model_path,
            task=args.task,
            n_eval=args.n_eval,
        )


if __name__ == '__main__':
    main()


# ============================================================================
# 使用示例
# ============================================================================

"""
# 1. 训练PPO (circle任务)
python baseline_ppo.py --mode train --task circle --timesteps 1000000 --n-envs 8

# 2. 评估训练好的模型
python baseline_ppo.py --mode eval --task circle --n-eval 100

# 3. 在多个任务上训练
for task in circle figure8 hover_wind; do
    python baseline_ppo.py --mode train --task $task --timesteps 1000000
done

# 4. 对比你的方法 vs PPO
python compare_methods.py --your-result results/circle_best.json \
                          --ppo-result ppo_baseline/circle/eval_logs/evaluations.npz
"""
