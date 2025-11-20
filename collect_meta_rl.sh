#!/usr/bin/env bash
# ==============================================================================
# Meta-RL 数据收集脚本
# ==============================================================================
# 功能：
#   收集 MCTS 超参数 sweep 数据（使用真实 Isaac Gym 奖励）
#
# 使用：
#   ./run_meta_rl.sh                      # 默认配置
#   COLLECT_RUNS=4 ./run_meta_rl.sh      # 收集 4 轮
#   FORCE_OVERWRITE=1 ./run_meta_rl.sh   # 清空旧数据重新开始
#
# 注意：
#   - 默认会自动追加到已有 CSV（不会覆盖）
#   - 训练请使用 ./train_meta_rl.sh
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================
# 环境检测
# ==============================
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  echo "[collect] 使用项目虚拟环境: $PYTHON_BIN"
else
  PYTHON_BIN="python3"
  echo "[collect] 使用系统 python3"
fi

# ==============================================================================
# ⚙️  可调参数区域（直接修改这里的数值）
# ==============================================================================
COLLECT_RUNS=1           # 收集轮数（每个配置跑 4 次不同随机轨迹）
COLLECT_ITERS=200        # 每个配置的训练轮数（⭐ 优化为 200 轮）
COLLECT_MCTS_SIMS=400    # 每轮的 MCTS 模拟次数（⭐ 优化为 400 次）
SAMPLE_INTERVAL=10       # 采样间隔：每 N 轮记录一次（10 = 产生 10 个时间步）
OUTPUT_CSV="results/mcts_tune/summary.csv"  # 输出 CSV 文件路径
PARALLEL_JOBS=4        # 并行进程数（先用 1 避免死机，稳定后再改为 4-8）

# ⚡ 当前配置（5×5×5×4 = 500 配置）：
# - 单进程（真实评估）: 500 × 100 × 3秒 = 41.7 小时
# - 4 进程并行: 41.7 ÷ 4 = 10.4 小时 ⭐
# - 8 进程并行: 41.7 ÷ 8 = 5.2 小时
# - 数据量: 500 × 2轮 × 10点 = 10,000 行（足够训练）
# ==============================================================================

# 支持环境变量覆盖（高级用法）
COLLECT_RUNS=${COLLECT_RUNS:-$COLLECT_RUNS}
COLLECT_ITERS=${COLLECT_ITERS:-$COLLECT_ITERS}
COLLECT_MCTS_SIMS=${COLLECT_MCTS_SIMS:-$COLLECT_MCTS_SIMS}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-$SAMPLE_INTERVAL}
OUTPUT_CSV=${OUTPUT_CSV:-$OUTPUT_CSV}
PARALLEL_JOBS=${PARALLEL_JOBS:-$PARALLEL_JOBS}

# 转换为绝对路径
if [[ "$OUTPUT_CSV" = /* ]]; then
  OUTPUT_ABS="$OUTPUT_CSV"
else
  OUTPUT_ABS="$REPO_ROOT/$OUTPUT_CSV"
fi

echo "=========================================="
echo "Meta-RL 数据收集"
echo "=========================================="
echo "配置参数："
echo "  - 收集轮数: $COLLECT_RUNS"
echo "  - 每个配置: $COLLECT_ITERS 轮训练"
echo "  - MCTS 模拟: $COLLECT_MCTS_SIMS 次/轮"
echo "  - 采样间隔: 每 $SAMPLE_INTERVAL 轮记录一次"
echo "  - 并行进程: $PARALLEL_JOBS 个"
echo "  - 输出文件: $OUTPUT_ABS"
echo "=========================================="

# 检查是否强制覆盖
if [[ -n "${FORCE_OVERWRITE:-}" ]]; then
  echo ""
  echo "⚠️  FORCE_OVERWRITE=1，将清空旧数据"
  rm -f "$OUTPUT_ABS"
fi

# 构建数据收集命令参数
collect_args=(
  --output "$OUTPUT_ABS"
  --python "$PYTHON_BIN"
  --total-iters "$COLLECT_ITERS"
  --mcts-sims "$COLLECT_MCTS_SIMS"
  --sample-interval "$SAMPLE_INTERVAL"
  --parallel "$PARALLEL_JOBS"
)

echo "  - 评估模式: 真实 Isaac Gym 评估（DummyEvaluator 已禁用）"

# 如果 CSV 已存在且非空，使用追加模式
if [[ -f "$OUTPUT_ABS" && -s "$OUTPUT_ABS" ]]; then
  echo ""
  echo "检测到已有数据，使用追加模式"
  collect_args+=(--append)
fi

# 运行数据收集（支持多轮）
for ((run=1; run<=COLLECT_RUNS; run++)); do
  echo ""
  echo "[$run/$COLLECT_RUNS] 开始第 $run 轮数据收集..."
  "$PYTHON_BIN" "$REPO_ROOT/scripts/meta_rl_collect.py" "${collect_args[@]}"
  
  # 第一轮后都是追加模式
  if [[ $run -eq 1 && ! " ${collect_args[*]} " =~ " --append " ]]; then
    collect_args+=(--append)
  fi
done

echo ""
echo "=========================================="
echo "✓ 数据收集完成！"
echo "数据文件: $OUTPUT_ABS"
echo "=========================================="
echo ""
echo "下一步：运行 ./train_meta_rl.sh 训练 RNN 控制器"

