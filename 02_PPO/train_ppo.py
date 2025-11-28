from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CRITICAL: 必须先导入 scg_vec_env（它会在 stable_baselines3 之前导入 Isaac Gym）
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from scg_vec_env import IsaacSCGVecEnv

# 现在可以安全导入 stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO with SCG reward on the Isaac quadrotor task")
    parser.add_argument("--task", default="figure8", choices=["figure8", "hover", "circle", "helix", "square"], help="reference trajectory")
    parser.add_argument("--duration", type=float, default=5.0, help="episode duration in seconds")
    parser.add_argument("--num-envs", type=int, default=32, help="parallel Isaac envs")
    parser.add_argument("--total-steps", type=int, default=400_000, help="total PPO environment steps")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout horizon")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="optimizer learning rate")
    parser.add_argument("--device", default="cuda:0", help="torch device for env + policy")
    parser.add_argument("--log-dir", default="results/ppo/figure8", help="logging/checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000, help="environment steps between checkpoints")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> IsaacSCGVecEnv:
    return IsaacSCGVecEnv(
        num_envs=args.num_envs,
        device=args.device,
        task=args.task,
        duration=args.duration,
    )


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_env = make_env(args)
    base_env.seed(args.seed)

    # 只归一化观测，不动 reward（保持 SCG 量纲）
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    tensorboard_log = log_dir / "tb"
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(tensorboard_log),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        max_grad_norm=0.5,
        ent_coef=0.0,
        vf_coef=0.5,
        device=args.device,
        seed=args.seed,
    )

    new_logger = configure(folder=str(log_dir), format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_interval // max(args.num_envs, 1), 1),
        save_path=str(log_dir),
        name_prefix="ppo_quadrotor",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    import time
    start_time = time.time()
    model.learn(total_timesteps=args.total_steps, callback=checkpoint_callback, progress_bar=True)
    elapsed = time.time() - start_time

    latest_path = log_dir / "ppo_quadrotor_latest"
    model.save(str(latest_path))
    env.close()

    steps_per_sec = args.total_steps / max(elapsed, 1e-6)
    sim_steps = args.total_steps  # 每个 env 步等价于一次 SCG 代价评估
    print("=" * 60)
    print("PPO Training Summary (Compute Budget)")
    print("=" * 60)
    print(f"Total env steps: {sim_steps}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Wall-clock time: {elapsed/3600:.2f} h")
    print(f"Throughput: {steps_per_sec:.0f} env-steps/s")
    print(f"Final checkpoint: {latest_path}.zip")


if __name__ == "__main__":
    main()
