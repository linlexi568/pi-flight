#!/bin/bash
################################################################################
# Pi-Flight 快速验证脚本 (10轮迭代)
# 用于快速验证训练系统是否正常工作
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

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                Pi-Flight 快速验证 (10 iterations)              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📋 测试配置:${NC}"
echo -e "  迭代数: 10"
echo -e "  MCTS模拟: 100次"
echo -e "  并行环境: 128"
echo -e "  Episode时长: 4秒"
echo -e "  预计用时: 8-10分钟"
echo ""

mkdir -p logs results

export DEBUG_STEPWISE=1

echo -e "${GREEN}🚀 开始测试...${NC}\n"

"$VENV_PYTHON" 01_pi_flight/train_online.py \
    --traj hover \
    --duration 4 \
    --total-iters 10 \
    --mcts-simulations 100 \
    --real-sim-frac 1.0 \
    --update-freq 5 \
    --train-steps-per-update 4 \
    --batch-size 64 \
    --isaac-num-envs 128 \
    --eval-replicas-per-program 2 \
    --min-steps-frac 0.3 \
    --reward-reduction mean \
    --save-path "$RESULT_FILE" \
    --use-fast-path \
    --debug-programs \
    --debug-programs-limit 24 \
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
