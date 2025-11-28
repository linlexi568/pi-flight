#!/usr/bin/env bash
# ==============================================================================
# PPO 训练启动脚本（禁止终端传参，直接修改下方常量即可）
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ==============================================================================
# 训练参数（按需修改）
# ==============================================================================
TASK="figure8"               # 轨迹类型：figure8 / hover / circle / helix
DURATION="5"                 # 每个 episode 的秒数

# 这一版是你之前跑出最好结果（≈ -12.39）的规模：
# 512 并行环境、总步数 1e7、每次 rollout 4096。
NUM_ENVS="512"               # 并行环境数量
TOTAL_STEPS="100000000"       # PPO 总交互步数
N_STEPS="4096"               # 每次更新的 rollout 长度
BATCH_SIZE="2048"            # PPO batch size（需整除 NUM_ENVS*N_STEPS）
LEARNING_RATE="1e-4"         # 学习率（保守稳定）
DEVICE="cuda:0"              # 训练设备
LOG_DIR="results/ppo/figure8_best"  # 日志与 checkpoint 目录（单独放 best 配置）
CHECKPOINT_INTERVAL="200000" # 保存间隔（环境步数）
SEED="0"                     # 随机种子
# ==============================================================================

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

mkdir -p "$LOG_DIR"

python3 02_PPO/train_ppo.py \
    --task "$TASK" \
    --duration "$DURATION" \
    --num-envs "$NUM_ENVS" \
    --total-steps "$TOTAL_STEPS" \
    --n-steps "$N_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --log-dir "$LOG_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --seed "$SEED"
