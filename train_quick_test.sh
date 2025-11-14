#!/bin/bash
################################################################################
# Pi-Flight 快速验证脚本 (单agent训练)
# 直接修改下面的参数变量即可调整配置，无需在终端传参
################################################################################

set -e

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/quick_test_${TIMESTAMP}.log"
RESULT_FILE="results/quick_test_${TIMESTAMP}.json"

# ========== 训练主循环参数 ==========
TOTAL_ITERS=10                 # 总迭代次数（快速验证用10，正式训练建议100-1000）
MCTS_SIMULATIONS=100           # 每次MCTS搜索的模拟次数
REAL_SIM_FRAC=1.0              # 真实仿真比例（1.0=全部真实仿真，0.0=全部NN估值）
UPDATE_FREQ=5                  # 每N次迭代执行一次NN参数更新
TRAIN_STEPS_PER_UPDATE=4       # 每次NN更新时从replay buffer采样训练N次
BATCH_SIZE=64                  # 每个训练step的batch大小

# ========== 无人机任务参数 ==========
TRAJ="hover"                   # 飞行轨迹：hover（悬停）、figure8（8字形）、circle（圆形）
DURATION=4                     # 单次飞行仿真时长（秒，快速验证用4秒）
ISAAC_NUM_ENVS=128             # Isaac Gym并行环境数（快速验证用128，正式训练建议512/1024）
EVAL_REPLICAS_PER_PROGRAM=2    # 每个程序评估时的重复次数（增加可提高评估稳定性）
MIN_STEPS_FRAC=0.3             # 最小步数比例（程序必须至少完成30%的飞行）
REWARD_REDUCTION="mean"        # 奖励聚合方式：mean（平均）、sum（求和）

# ========== 调试与保存参数 ==========
USE_FAST_PATH="--use-fast-path"           # 使用快速路径优化（建议保持）
DEBUG_PROGRAMS="--debug-programs"         # 打印程序详情（调试用）
DEBUG_PROGRAMS_LIMIT=24                   # 打印前N个程序

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            Pi-Flight 快速验证 ($TOTAL_ITERS iterations)              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📋 当前配置:${NC}"
echo -e "  迭代数: $TOTAL_ITERS"
echo -e "  MCTS模拟: $MCTS_SIMULATIONS 次"
echo -e "  并行环境: $ISAAC_NUM_ENVS"
echo -e "  Episode时长: $DURATION 秒"
echo -e "  轨迹类型: $TRAJ"
echo ""

mkdir -p logs results

export DEBUG_STEPWISE=1

echo -e "${GREEN}🚀 开始训练...${NC}\n"

"$VENV_PYTHON" 01_pi_flight/train_online.py \
    --traj $TRAJ \
    --duration $DURATION \
    --total-iters $TOTAL_ITERS \
    --mcts-simulations $MCTS_SIMULATIONS \
    --real-sim-frac $REAL_SIM_FRAC \
    --update-freq $UPDATE_FREQ \
    --train-steps-per-update $TRAIN_STEPS_PER_UPDATE \
    --batch-size $BATCH_SIZE \
    --isaac-num-envs $ISAAC_NUM_ENVS \
    --eval-replicas-per-program $EVAL_REPLICAS_PER_PROGRAM \
    --min-steps-frac $MIN_STEPS_FRAC \
    --reward-reduction $REWARD_REDUCTION \
    --save-path "$RESULT_FILE" \
    $USE_FAST_PATH \
    $DEBUG_PROGRAMS \
    --debug-programs-limit $DEBUG_PROGRAMS_LIMIT \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 测试完成!${NC}"
    echo ""
    echo -e "查看奖励进展:"
    grep -E '\[Iter [0-9]+\].*完成.*奖励:' "$LOG_FILE" || true
    echo ""
    echo -e "${YELLOW}如果看到奖励提升,可运行完整训练: ./train_full.sh${NC}"
else
    echo -e "\n${RED}❌ 测试失败,退出代码: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
