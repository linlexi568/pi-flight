# PPO Baseline for SCG-Aligned Quadrotor Tracking

This module hosts a clean PPO training pipeline that interacts with the existing Isaac Gym quadrotor environment and the exact safe-control-gym (SCG) reward used by π-Flight. It is designed to produce policies that can compete with or surpass the PID/LQR baselines under the same `reward_true` metric.

## Features

- **SCG Reward:** Rewards are computed exclusively via `SCGExactRewardCalculator`, so PPO optimizes the same cost used for π-Flight programs and PID/LQR baselines.
- **VecEnv Wrapper:** `IsaacSCGVecEnv` exposes the Isaac Gym simulator to Stable Baselines 3 using the `[u_fz, u_tx, u_ty, u_tz]` thrust/torque channels.
- **Training & Evaluation Utilities:** Scripts to train PPO policies and evaluate them against the SCG reward, optionally comparing with PID/LQR JSON summaries.

## Quick Start

> Ensure Isaac Gym is installed and the root `requirements.txt` (which already includes `stable_baselines3`) is installed in your virtual environment.

```bash
# Train PPO on the 5s figure-8 task with 32 parallel envs
python 02_PPO/train_ppo.py \
  --task figure8 \
  --duration 5 \
  --num-envs 32 \
  --total-steps 400000 \
  --log-dir results/ppo/figure8

# Evaluate a saved checkpoint (deterministic policy)
python 02_PPO/eval_ppo.py \
  --model results/ppo/figure8/ppo_quadrotor_latest.zip \
  --task figure8 \
  --duration 5 \
  --episodes 10
```

The evaluation script reports the mean and standard deviation of `reward_true` (negative SCG cost). Compare these numbers against `results/baseline/<task>_baseline.json` or `results/scg_aligned/*` to verify PPO outperforms PID/LQR.

## File Layout

- `scg_vec_env.py` — Stable-Baselines VecEnv wrapper around `IsaacGymDroneEnv` with SCG reward.
- `train_ppo.py` — Training entrypoint for PPO, includes checkpointing and TensorBoard logging hooks.
- `eval_ppo.py` — Deterministic roll-out script producing SCG rewards for qualitative comparison.

