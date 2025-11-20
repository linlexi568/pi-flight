#!/usr/bin/env bash
# ==============================================================================
# π-Flight 训练启动脚本（仅 Train 模式）
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================
# 环境检测 & 显存优化参数
# ==============================
# Ranking训练显存控制：每批最多处理的图数量（降低此值可减少OOM风险）
# 推荐值：4（默认），高显存卡可提升至8或16；6GB以下显卡建议2或1
export RANKING_GNN_CHUNK=4

# PyTorch 内存碎片控制：避免 reserved >> allocated 导致的 OOM（可按需覆盖）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

TRAIN_ENTRY="${REPO_ROOT}/01_pi_flight/train_online.py"
RESULT_DIR="${REPO_ROOT}/results"

# ==============================
# 训练参数配置
# ==============================
TRAIN_TOTAL_ITERS=100
TRAIN_MCTS_SIMS=150               # 🚀 200次模拟，生成~200个程序
TRAIN_UPDATE_FREQ=8               # NN更新频率：每8轮迭代更新一次
TRAIN_ISAAC_ENVS=4096             # 并行环境数量（恢复到4096，之前成功运行过）
TRAIN_MCTS_LEAF_BATCH=12          # MCTS 叶节点批量大小
ENABLE_ASYNC_TRAINING=true       # 异步训练模式
ASYNC_UPDATE_INTERVAL=0.15       # 异步训练步之间的时间间隔(秒)
ASYNC_MAX_STEPS_PER_ITER=2       # 每轮允许的最大异步训练步数

# 高级优化开关
USE_RANKING=false                # 是否启用Ranking网络训练（设为false完全关闭ranking）
ENABLE_RANKING_MCTS_BIAS=false  # Ranking网络对MCTS子节点先验加权
RANKING_BIAS_BETA=0.3            # Ranking bias强度
ENABLE_VALUE_HEAD=false           # 启用Value头辅助训练
ENABLE_RANKING_REWEIGHT=false    # 用Ranking score重新加权policy target
RANKING_REWEIGHT_BETA=0.2       # Ranking reweight强度

TRAIN_SAVE_PATH="$RESULT_DIR/online_best_program.json"
TRAIN_EXTRA_ARGS=(
  "--zero-action-penalty" "5.0"
  "--zero-action-penalty-decay" "0.98"
  "--zero-action-penalty-min" "1.0"
  "--reward-reduction" "mean"
  "--eval-replicas-per-program" "1"  # ✅ 确定性评估：无随机性，1次足够（3副本浪费3x时间）
  "--use-fast-path"  # ✅ 启用CUDA加速路径
)

# GPU 表达式执行开关（默认启用，可通过环境变量覆盖以回退到CPU执行）
: "${ENABLE_GPU_EXPRESSION:=true}"

# Meta-RL 开关
USE_META_RL=false
META_CKPT="meta_rl/checkpoints/meta_policy.pt"

# 启发式调参
HEURISTIC_ROOT_EPS_INIT=0.25
HEURISTIC_ROOT_EPS_FINAL=0.10
HEURISTIC_ROOT_ALPHA_INIT=0.30
HEURISTIC_ROOT_ALPHA_FINAL=0.20
HEURISTIC_DECAY_WINDOW=350

# GNN 架构
GNN_STRUCTURE_HIDDEN=192
GNN_STRUCTURE_LAYERS=4
GNN_STRUCTURE_HEADS=6
GNN_FEATURE_LAYERS=4
GNN_FEATURE_HEADS=8
GNN_DROPOUT=0.2

# NN 训练
NN_BATCH_SIZE=128
NN_LEARNING_RATE=0.001
NN_REPLAY_CAPACITY=50000

# 环境
TRAJECTORY="figure8"
DURATION=8 
# 🔥 奖励 profile - 可选：
#   safety_first           - 保守、平滑、节能（强调安全性）
#   tracking_first         - 激进跟踪、允许大动作（强调跟踪精度）
#   balanced               - 折中方案（综合平衡）
#   robustness_stability   - 鲁棒性+稳定性优先（之前的主实验，强调抗扰动、增益稳定）
#   control_law_discovery  - 同 robustness_stability（别名，向后兼容）
#   smooth_control         - 平滑控制优先
#   balanced_smooth        - 平衡平滑
REWARD_PROFILE="balanced"

# 先验参数（结构/稳定性偏好）
PRIOR_PROFILE="structure_stability"                # 可选: none, structure, structure_stability
STRUCTURE_PRIOR_WEIGHT=0.35          # 结构先验权重（推荐 0.35）
STABILITY_PRIOR_WEIGHT=0.2          # 稳定性先验权重（推荐 0.20）

ENABLE_BAYESIAN_TUNING=true        # 🚀 真实GP-UCB贝叶斯优化
BO_BATCH_SIZE=10                      # BO每轮候选数（优化：从15减至10，减少33%评估）
BO_ITERATIONS=4                       # BO迭代轮数（优化：从4减至3，总候选30个/程序）
BO_PARAM_RANGE_MIN=-2.0              # 参数搜索下界
BO_PARAM_RANGE_MAX=2.0               # 参数搜索上界

# 🔬 实验开关：禁用GNN（使用均匀先验）对比性能
DISABLE_GNN=true                   # true=禁用GNN训练和推理，仅用均匀先验

# ==============================
# 启动训练
# ==============================
if [[ ! -f "$TRAIN_ENTRY" ]]; then
  echo "[run.sh] 训练入口脚本缺失: $TRAIN_ENTRY" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[run.sh] Python 可执行文件未找到: $PYTHON_BIN" >&2
  exit 1
fi

echo "============================================================"
echo "π-Flight 训练启动"
echo "============================================================"
echo "总轮数: ${TRAIN_TOTAL_ITERS}"
echo "MCTS模拟: ${TRAIN_MCTS_SIMS} 次/轮"
echo "并行环境: ${TRAIN_ISAAC_ENVS}"
echo "输出路径: $TRAIN_SAVE_PATH"
echo "============================================================"

if [[ "$USE_META_RL" == "true" ]]; then
  echo "调参模式: Meta-RL"
else
  echo "调参模式: 启发式退火"
fi

mkdir -p "$(dirname "$TRAIN_SAVE_PATH")"

cmd=(
  "$PYTHON_BIN" "$TRAIN_ENTRY"
  --total-iters "$TRAIN_TOTAL_ITERS"
  --mcts-simulations "$TRAIN_MCTS_SIMS"
  --mcts-leaf-batch-size "$TRAIN_MCTS_LEAF_BATCH"
  --update-freq "$TRAIN_UPDATE_FREQ"
  --isaac-num-envs "$TRAIN_ISAAC_ENVS"
  --save-path "$TRAIN_SAVE_PATH"
)

if [[ "$ENABLE_ASYNC_TRAINING" == "true" ]]; then
  cmd+=(--async-training --async-update-interval "$ASYNC_UPDATE_INTERVAL")
  if [[ -n "${ASYNC_MAX_STEPS_PER_ITER}" ]]; then
    cmd+=(--async-max-steps-per-iter "$ASYNC_MAX_STEPS_PER_ITER")
  fi
  echo "✓ 异步训练已启用 (interval=${ASYNC_UPDATE_INTERVAL}s, maxSteps=${ASYNC_MAX_STEPS_PER_ITER:-inf})"
fi

if [[ "$ENABLE_GPU_EXPRESSION" == "false" ]]; then
  cmd+=(--disable-gpu-expression)
  echo "✓ GPU 表达式执行已禁用（使用CPU回退）"
else
  echo "✓ GPU 表达式执行启用"
fi

if [[ "$ENABLE_RANKING_MCTS_BIAS" == "true" ]]; then
  cmd+=(--enable-ranking-mcts-bias --ranking-bias-beta "$RANKING_BIAS_BETA")
  echo "✓ Ranking MCTS Bias (beta=$RANKING_BIAS_BETA)"
fi

# Ranking 主开关
if [[ "$USE_RANKING" == "false" ]]; then
  cmd+=(--use-ranking false)
  echo "✓ Ranking网络已完全关闭（纯主线模式）"
fi

if [[ "$ENABLE_VALUE_HEAD" == "true" ]]; then
  cmd+=(--enable-value-head)
  echo "✓ Value头辅助训练"
fi

if [[ "$ENABLE_RANKING_REWEIGHT" == "true" ]]; then
  cmd+=(--enable-ranking-reweight --ranking-reweight-beta "$RANKING_REWEIGHT_BETA")
  echo "✓ Ranking重加权 (beta=$RANKING_REWEIGHT_BETA)"
fi

if [[ "$USE_META_RL" == "true" ]]; then
  if [[ ! -f "$META_CKPT" ]]; then
    echo "Meta-RL 模型未找到: $META_CKPT" >&2
    exit 1
  fi
  cmd+=(--use-meta-rl --meta-rl-checkpoint "$META_CKPT")
else
  cmd+=(
    --root-dirichlet-eps-init "$HEURISTIC_ROOT_EPS_INIT"
    --root-dirichlet-eps-final "$HEURISTIC_ROOT_EPS_FINAL"
    --root-dirichlet-alpha-init "$HEURISTIC_ROOT_ALPHA_INIT"
    --root-dirichlet-alpha-final "$HEURISTIC_ROOT_ALPHA_FINAL"
    --heuristic-decay-window "$HEURISTIC_DECAY_WINDOW"
  )
fi

cmd+=(
  --gnn-structure-hidden "$GNN_STRUCTURE_HIDDEN"
  --gnn-structure-layers "$GNN_STRUCTURE_LAYERS"
  --gnn-structure-heads "$GNN_STRUCTURE_HEADS"
  --gnn-feature-layers "$GNN_FEATURE_LAYERS"
  --gnn-feature-heads "$GNN_FEATURE_HEADS"
  --gnn-dropout "$GNN_DROPOUT"
  --batch-size "$NN_BATCH_SIZE"
  --learning-rate "$NN_LEARNING_RATE"
  --replay-capacity "$NN_REPLAY_CAPACITY"
  --traj "$TRAJECTORY"
  --duration "$DURATION"
  --reward-profile "$REWARD_PROFILE"
  --prior-profile "$PRIOR_PROFILE"
  --structure-prior-weight "$STRUCTURE_PRIOR_WEIGHT"
  --stability-prior-weight "$STABILITY_PRIOR_WEIGHT"
)

# 🔥 贝叶斯优化调参参数
if [[ "$ENABLE_BAYESIAN_TUNING" == "true" ]]; then
  cmd+=(
    --enable-bayesian-tuning
    --bo-batch-size "$BO_BATCH_SIZE"
    --bo-iterations "$BO_ITERATIONS"
    --bo-param-range-min "$BO_PARAM_RANGE_MIN"
    --bo-param-range-max "$BO_PARAM_RANGE_MAX"
  )
  echo "✓ 贝叶斯优化调参已启用 (batch=$BO_BATCH_SIZE, iters=$BO_ITERATIONS)"
fi

if [[ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${TRAIN_EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "开始训练..."
echo "============================================================"

# 自动生成带时间戳的日志文件
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${REPO_ROOT}/logs/train_${TIMESTAMP}.log"
mkdir -p "${REPO_ROOT}/logs"

echo "日志将保存到: $LOG_FILE"
echo ""

# 使用 tee 同时输出到终端和日志文件，确保捕获所有 stdout/stderr
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"

# 保存退出码
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "✅ 训练成功完成"
  echo "============================================================"
else
  echo ""
  echo "============================================================"
  echo "❌ 训练异常退出 (退出码: $TRAIN_EXIT_CODE)"
  echo "完整日志已保存: $LOG_FILE"
  echo "============================================================"
fi

exit $TRAIN_EXIT_CODE
