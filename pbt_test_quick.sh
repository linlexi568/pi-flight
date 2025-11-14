#!/bin/bash
# PBT快速测试脚本 - 4 agents × 100 iterations

cd "$(dirname "$0")"

echo "=========================================="
echo "PBT快速测试"
echo "配置: 4 agents, 100 iters, 20轮PBT调度"
echo "=========================================="

python 01_pi_flight/train_pbt.py \
    --n-agents 4 \
    --pbt-interval 20 \
    --exploit-threshold 0.25 \
    --shared-encoder \
    --total-iters 100 \
    --update-freq 10 \
    --train-steps-per-update 5 \
    --batch-size 32 \
    --replay-capacity 5000 \
    --hidden-channels 64 \
    --num-gnn-layers 2 \
    --traj circle \
    --duration 5 \
    --isaac-num-envs 128 \
    --reward-profile control_law_discovery \
    --save-path results/pbt_test.json \
    --save-freq 50 \
    "$@"
