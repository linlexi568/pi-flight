"""A/B Benchmark for GNN v1 vs v2

运行方式 (示例):

  /home/linlexi/桌面/pi-flight/.venv/bin/python ab_benchmark.py \
    --iters 120 --mcts 300 --traj figure8 --isaac-num-envs 128

脚本会顺序运行 v1 与 v2 两次短训练 (使用在线训练主循环, 减少迭代和并行规模加速) 并输出摘要:
- 最佳奖励
- 收敛曲线的前若干点 (每10轮)
- 参数量对比

注意: 为快速比较仅适合初期收敛趋势评估, 不代表长期稳定性能。
"""
from __future__ import annotations
import argparse, time, json, random, os, sys, pathlib
import numpy as np
import torch

# 目录处理
ROOT = pathlib.Path(__file__).resolve().parent
PKG = ROOT / '01_pi_flight'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from train_online import OnlineTrainer
from argparse import Namespace


def run_short_training(nn_version: str, base_args, iters: int, mcts: int, seed: int):
    # 构造最小必要参数对象，不调用训练脚本的命令行解析避免冲突
    args = Namespace(
        total_iters=iters,
        mcts_simulations=mcts,
        update_freq=max(10, iters // 12),
        train_steps_per_update=5,
        batch_size=128,
        replay_capacity=20000,
        use_gnn=True,
        nn_version=nn_version,
        nn_hidden=256,
        learning_rate=1e-3,
        value_loss_weight=0.5,
        exploration_weight=1.4,
        puct_c=1.5,
        max_depth=20,
        real_sim_frac=0.8,
        traj=base_args.traj,
        duration=base_args.duration,
        isaac_num_envs=base_args.isaac_num_envs,
        eval_replicas_per_program=1,
        min_steps_frac=0.0,
        reward_reduction='sum',
    use_fast_path=False,
    use_dummy_eval=True,
        save_path=f"01_pi_flight/results/ab_best_program_{nn_version}.json",
        checkpoint_freq=10**9,
        warm_start=None,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    trainer = OnlineTrainer(args)

    rewards = []
    best = -1e9
    for i in range(args.total_iters):
        children, visit_counts = trainer.mcts_search(trainer._generate_random_program(), args.mcts_simulations)
        if not children:
            rewards.append(best)
            continue
        # choose best
        idx = int(np.argmax(visit_counts))
        prog = children[idx].program
        reward = trainer.evaluator.evaluate_single(prog)
        if reward > best:
            best = reward
        rewards.append(best)
    return {
        'nn_version': nn_version,
        'best_reward': best,
        'curve': rewards,
        'param_count': sum(p.numel() for p in trainer.nn_model.parameters())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--short-iters', type=int, default=120, help='每个模型短训练迭代数')
    ap.add_argument('--short-mcts', type=int, default=300, help='每迭代MCTS模拟数')
    ap.add_argument('--traj', type=str, default='figure8')
    ap.add_argument('--duration', type=int, default=6)
    ap.add_argument('--isaac-num-envs', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print("==== A/B Benchmark 开始 ====")
    print(f"配置: iters={args.short_iters}, mcts={args.short_mcts}, traj={args.traj}, envs={args.isaac_num_envs}")

    t0 = time.time()
    res_v1 = run_short_training('v1', args, args.short_iters, args.short_mcts, args.seed)
    t1 = time.time()
    res_v2 = run_short_training('v2', args, args.short_iters, args.short_mcts, args.seed)
    t2 = time.time()

    def summarize(r):
        curve = r['curve']
        points = [curve[i] for i in range(0, len(curve), max(1, len(curve)//10))]
        return points

    print("\n==== 结果摘要 ====")
    print(f"v1 最佳奖励: {res_v1['best_reward']:.4f} | 参数量: {res_v1['param_count']:,}")
    print(f"v1 收敛片段: {summarize(res_v1)}")
    print(f"耗时: {(t1 - t0):.1f}s")
    print(f"v2 最佳奖励: {res_v2['best_reward']:.4f} | 参数量: {res_v2['param_count']:,}")
    print(f"v2 收敛片段: {summarize(res_v2)}")
    print(f"耗时: {(t2 - t1):.1f}s")

    diff = res_v2['best_reward'] - res_v1['best_reward']
    print(f"\nΔ(best_reward v2 - v1) = {diff:.4f}")
    if diff > 0.0:
        print("✅ v2 在此短基准中表现更好，建议迁移主训练脚本到 v2")
    else:
        print("⚠️ v2 尚未在短基准中超越 v1，可增加迭代或调参再观察")

    # 保存 JSON
    out = {
        'config': vars(args),
        'v1': res_v1,
        'v2': res_v2,
        'delta_best_reward': diff,
        'total_time_s': t2 - t0
    }
    with open('01_pi_flight/results/ab_summary.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n📄 已保存结果到 01_pi_flight/results/ab_summary.json")

if __name__ == '__main__':
    main()
