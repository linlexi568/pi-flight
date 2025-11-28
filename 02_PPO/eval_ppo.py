from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

# 先导入 scg_vec_env（内部会优先导入 Isaac Gym，再导入 torch）
from scg_vec_env import IsaacSCGVecEnv

# 再导入 stable_baselines3，避免其提前导入 torch
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint under the SCG reward")
    parser.add_argument("--model", required=True, help="path to the saved PPO .zip file")
    parser.add_argument("--task", default="figure8", choices=["figure8", "hover", "circle", "helix", "square"], help="reference trajectory")
    parser.add_argument("--duration", type=float, default=5.0, help="episode duration in seconds")
    parser.add_argument("--episodes", type=int, default=10, help="number of evaluation episodes")
    parser.add_argument("--device", default="cuda:0", help="torch device for env + policy")
    parser.add_argument("--num-envs", type=int, default=4, help="parallel envs for batched evaluation")
    parser.add_argument("--output", help="optional JSON file to store summary stats")
    return parser.parse_args()


def rollout(model: PPO, env: IsaacSCGVecEnv, episodes: int) -> List[float]:
    obs = env.reset()
    per_env_returns = np.zeros(env.num_envs, dtype=np.float32)
    episode_returns: List[float] = []

    while len(episode_returns) < episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(actions)
        per_env_returns += rewards
        for idx, done in enumerate(dones):
            if done:
                episode_returns.append(float(per_env_returns[idx]))
                per_env_returns[idx] = 0.0
                if len(episode_returns) >= episodes:
                    break
    return episode_returns[:episodes]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    env = IsaacSCGVecEnv(
        num_envs=args.num_envs,
        device=args.device,
        task=args.task,
        duration=args.duration,
    )

    model = PPO.load(str(model_path), device=args.device)
    returns = rollout(model, env, args.episodes)
    env.close()

    returns_arr = np.asarray(returns, dtype=np.float32)
    mean_reward = float(returns_arr.mean())
    std_reward = float(returns_arr.std(ddof=0))

    print("=" * 60)
    print("PPO Evaluation Summary")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Task / duration: {args.task} / {args.duration}s")
    print(f"Episodes: {args.episodes}")
    print(f"reward_true mean: {mean_reward:.2f} | std: {std_reward:.2f}")
    print("Returns:")
    for idx, ret in enumerate(returns):
        print(f"  Episode {idx+1:02d}: {ret:.2f}")

    if args.output:
        summary = {
            "model": str(model_path),
            "task": args.task,
            "duration": args.duration,
            "episodes": args.episodes,
            "mean_reward_true": mean_reward,
            "std_reward_true": std_reward,
            "returns": returns,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
