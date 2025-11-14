#!/bin/bash
# Population-Based Training启动脚本
# 16-agent并行训练，自动调节MCTS参数

cd "$(dirname "$0")"

python 01_pi_flight/train_pbt.py \
    --n-agents 16 \
    --pbt-interval 50 \
    --exploit-threshold 0.25 \
    --shared-encoder \
    --total-iters 5000 \
    --update-freq 50 \
    --train-steps-per-update 10 \
    --batch-size 64 \
    --replay-capacity 20000 \
    --hidden-channels 128 \
    --num-gnn-layers 3 \
    --traj figure8 \
    --duration 10 \
    --isaac-num-envs 512 \
    --reward-profile control_law_discovery \
    --save-path results/pbt_best_program.json \
    --save-freq 200 \
    "$@"
