#!/usr/bin/env bash
# ==============================================================================
# Meta-RL 控制器单独训练脚本
# ==============================================================================
# 功能：
#   只训练 RNN 控制器，不进行数据收集
#   假设已经有了 results/mcts_tune/summary.csv
#
# 使用：
#   ./train_meta_rl.sh
#   ./train_meta_rl.sh --epochs 200 --batch-size 64
#
# 注意：
#   - 如果没有 summary.csv，请先运行 ./run_meta_rl.sh 或手动收集数据
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================
# 环境检测：优先使用项目虚拟环境
# ==============================
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  echo "[train] 使用项目虚拟环境: $PYTHON_BIN"
else
  PYTHON_BIN="python3"
  echo "[train] 未找到 .venv，使用系统 python3"
fi

# ==============================================================================
# ⚙️  可调参数区域（直接修改这里的数值）
# ==============================================================================
SUMMARY_CSV="results/mcts_tune/summary.csv"      # 输入：数据收集的 CSV 文件
OUTPUT_CKPT="meta_rl/results/meta_policy.pt" # 输出：训练好的模型

# RNN 训练超参数
EPOCHS=300               # 训练轮数（推荐：50-200）
BATCH_SIZE=256           # 批量大小（推荐：64-256）
LEARNING_RATE=1e-3       # 学习率（推荐：1e-4 到 5e-4）
WINDOW=16                # 时序窗口长度（推荐：10-20）

# RNN 网络结构
HIDDEN_DIM=256           # 隐藏层维度（推荐：64-256）
NUM_LAYERS=3             # RNN 层数（推荐：1-3）
# ==============================================================================

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --summary-csv)
      SUMMARY_CSV="$2"
      shift 2
      ;;
    --output)
      OUTPUT_CKPT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --window)
      WINDOW="$2"
      shift 2
      ;;
    --hidden-dim)
      HIDDEN_DIM="$2"
      shift 2
      ;;
    --num-layers)
      NUM_LAYERS="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
用法：./train_meta_rl.sh [选项]

选项：
  --summary-csv PATH    输入的 CSV 数据文件（默认：results/mcts_tune/summary.csv）
  --output PATH         输出的模型 checkpoint（默认：meta_rl/checkpoints/meta_policy.pt）
  --epochs N            训练轮数（默认：150）
  --batch-size N        批量大小（默认：128）
  --lr VALUE            学习率（默认：2e-4）
  --window N            时序窗口长度（默认：16）
  --hidden-dim N        RNN 隐藏层维度（默认：128）
  --num-layers N        RNN 层数（默认：2）
  -h, --help            显示此帮助信息

示例：
  ./train_meta_rl.sh
  ./train_meta_rl.sh --epochs 200 --batch-size 64
  ./train_meta_rl.sh --summary-csv my_data.csv --output my_model.pt
EOF
      exit 0
      ;;
    *)
      echo "未知参数：$1"
      echo "使用 --help 查看帮助"
      exit 1
      ;;
  esac
done

# 转换为绝对路径
if [[ "$SUMMARY_CSV" = /* ]]; then
  SUMMARY_ABS="$SUMMARY_CSV"
else
  SUMMARY_ABS="$REPO_ROOT/$SUMMARY_CSV"
fi

if [[ "$OUTPUT_CKPT" = /* ]]; then
  OUTPUT_ABS="$OUTPUT_CKPT"
else
  OUTPUT_ABS="$REPO_ROOT/$OUTPUT_CKPT"
fi

# 检查输入文件是否存在
if [[ ! -f "$SUMMARY_ABS" ]]; then
  echo "❌ 错误：找不到输入文件 $SUMMARY_ABS"
  echo "请先运行数据收集：./run_meta_rl.sh"
  exit 1
fi

# 显示配置信息
echo "=========================================="
echo "Meta-RL 控制器训练"
echo "=========================================="
echo "输入数据：$SUMMARY_ABS"
echo "输出模型：$OUTPUT_ABS"
echo "训练参数："
echo "  - 轮数 (epochs): $EPOCHS"
echo "  - 批量 (batch_size): $BATCH_SIZE"
echo "  - 学习率 (lr): $LEARNING_RATE"
echo "  - 窗口 (window): $WINDOW"
echo "  - 隐藏维度 (hidden_dim): $HIDDEN_DIM"
echo "  - RNN 层数 (num_layers): $NUM_LAYERS"
echo "=========================================="
echo ""

# 运行训练
"$PYTHON_BIN" "$REPO_ROOT/scripts/meta_rl_train.py" \
  --summary-csv "$SUMMARY_ABS" \
  --output-checkpoint "$OUTPUT_ABS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --window "$WINDOW" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS"

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
