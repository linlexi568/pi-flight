#!/bin/bash
# Training script for PPO baseline on all tasks

set -e  # Exit on error

echo "=========================================="
echo "PPO Baseline Training - All Tasks"
echo "=========================================="

# Configuration
TIMESTEPS=1000000
N_ENVS=8
SAVE_DIR="./02_PPO/results"

# Create results directory
mkdir -p "$SAVE_DIR"

# Tasks to train
TASKS=("circle" "figure8" "hover_wind")

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "===================="
    echo "Training task: $TASK"
    echo "===================="
    
    python baseline_ppo.py \
        --mode train \
        --task "$TASK" \
        --timesteps $TIMESTEPS \
        --n-envs $N_ENVS \
        --save-dir "$SAVE_DIR"
    
    echo ""
    echo "Task $TASK completed!"
    echo ""
done

echo "=========================================="
echo "All training completed!"
echo "=========================================="

# Optional: Run evaluation
echo ""
read -p "Run evaluation on all models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for TASK in "${TASKS[@]}"; do
        echo "Evaluating $TASK..."
        python baseline_ppo.py \
            --mode eval \
            --task "$TASK" \
            --model-path "$SAVE_DIR/$TASK/final_model" \
            --n-eval 100
    done
fi
