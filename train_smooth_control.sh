#!/bin/bash

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# 检查虚拟环境
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}❌ 虚拟环境不存在: $VENV_PYTHON${NC}"
    echo "请先运行: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

################################################################################
# 🔧 训练参数配置 - 平滑控制优先模式
################################################################################
CFG_ITERS=150                  # 总迭代数（测试跑150轮）
CFG_SIMS=800                   # 每轮MCTS模拟次数
CFG_ENVS=1024                  # 并行环境数量
CFG_DURATION=6                  # Episode时长(秒)
CFG_TRAJ="circle"                # 任务: hover/circle/figure8
CFG_EVAL_REPLICAS=3             # 每程序评估次数
CFG_UPDATE_FREQ=10               # 每N轮更新GNN
CFG_TRAIN_STEPS=10              # 每次更新训练步数
CFG_BATCH_SIZE=128              # 批大小
CFG_MIN_STEPS_FRAC=0.3        # 最短步数比例 
CFG_LR=1e-3                   # 学习率
CFG_CHECKPOINT_FREQ=40       # Checkpoint保存频率
CFG_REWARD_REDUCTION="mean"   # 奖励聚合: sum/mean
# 🔥 奖励配置：平衡型（不奔着PID，强调可部署性）
CFG_REWARD_PROFILE="balanced_smooth"  # 平衡平滑/控制代价/响应/物理约束

################################################################################
# 📌 说明:
# 使用 balanced_smooth 奖励配置，强调：
# - smoothness_jerk: 0.80 (鼓励平滑，但不过度抑制探索)
# - control_effort: 0.60 (限制大幅控制变化，保留响应)
# - high_freq: 0.80 (抑制高频振荡，兼顾灵活性)
# 适用于需要“不过度平滑 + 可部署 + 合理跟踪/响应”的控制策略场景
################################################################################

# 创建目录
mkdir -p logs results 01_pi_flight/results

# 生成时间戳文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/smooth_${CFG_ITERS}iters_${TIMESTAMP}.log"
RESULT_FILE="01_pi_flight/results/smooth_${CFG_ITERS}iters_${TIMESTAMP}.json"

# 显示配置
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Pi-Flight 平滑控制训练 (${CFG_ITERS} 轮) + Ranking NN         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📋 训练配置:${NC}"
echo -e "  模式: AlphaZero + Ranking NN"
echo -e "  任务: ${CFG_TRAJ}"
echo -e "  迭代: ${CFG_ITERS} 轮"
echo -e "  MCTS: ${CFG_SIMS} 次模拟/轮"
echo -e "  环境: ${CFG_ENVS} 个并行"
echo -e "  时长: ${CFG_DURATION} 秒/episode"
echo -e "  评估: ${CFG_EVAL_REPLICAS} 次/程序"
echo -e "  学习率: ${CFG_LR}"
echo -e "  Ranking: ✅ 启用 (lr=1e-3, 零动作惩罚=2.0→0.1)"
echo -e "  🔥 奖励配置: ${CFG_REWARD_PROFILE} (强调平滑度和控制代价)"
echo ""
echo -e "${YELLOW}奖励权重详情:${NC}"
echo -e "  smoothness_jerk: 1.20 (k=0.65) - 🔥 高权重抑制加加速度"
echo -e "  control_effort:  0.85 (k=0.35) - 🔥 高权重惩罚控制变化"
echo -e "  high_freq:       1.00 (k=3.2)  - 🔥 强抑制高频振荡"
echo -e "  position_rmse:   0.70 (k=0.9)  - 适度降低跟踪精度要求"
echo -e "  settling_time:   0.90 (k=1.1)  - 保持鲁棒性关注"
echo -e "  saturation:      1.10 (k=1.3)  - 严格惩罚饱和"
echo ""
echo -e "${YELLOW}💾 输出:${NC}"
echo -e "  日志: $LOG_FILE"
echo -e "  结果: $RESULT_FILE"
echo ""

# 自动启动(nohup兼容)
echo -e "${GREEN}🚀 启动训练...${NC}"
echo ""

# 启动训练
export DEBUG_STEPWISE=1

"$VENV_PYTHON" 01_pi_flight/train_online.py \
    --use-fast-path \
    --traj "${CFG_TRAJ}" \
    --duration "${CFG_DURATION}" \
    --total-iters "${CFG_ITERS}" \
    --mcts-simulations "${CFG_SIMS}" \
    --update-freq "${CFG_UPDATE_FREQ}" \
    --train-steps-per-update "${CFG_TRAIN_STEPS}" \
    --batch-size "${CFG_BATCH_SIZE}" \
    --isaac-num-envs "${CFG_ENVS}" \
    --eval-replicas-per-program "${CFG_EVAL_REPLICAS}" \
    --min-steps-frac "${CFG_MIN_STEPS_FRAC}" \
    --reward-reduction "${CFG_REWARD_REDUCTION}" \
    --reward-profile "${CFG_REWARD_PROFILE}" \
    --learning-rate "${CFG_LR}" \
    --checkpoint-freq "${CFG_CHECKPOINT_FREQ}" \
    --ranking-lr 1e-3 \
    --ranking-blend-init 0.3 \
    --ranking-blend-max 0.8 \
    --ranking-blend-warmup 100 \
    --zero-action-penalty 2.0 \
    --zero-action-penalty-decay 0.95 \
    --zero-action-penalty-min 0.1 \
    --save-path "$RESULT_FILE" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# 显示结果
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 训练完成!${NC}"
    echo -e "${BLUE}📊 结果: $RESULT_FILE${NC}"
    echo -e "${BLUE}📄 日志: $LOG_FILE${NC}"
    echo ""
    echo -e "${YELLOW}🔍 预期效果:${NC}"
    echo -e "  ✓ 更平滑的轨迹 (低 jerk)"
    echo -e "  ✓ 更小的控制输出变化"
    echo -e "  ✓ 低高频振荡"
    echo -e "  ✓ 物理可实现的控制策略"
else
    echo -e "${RED}❌ 训练失败 (退出码: $EXIT_CODE)${NC}"
    echo -e "${YELLOW}查看日志: $LOG_FILE${NC}"
    exit $EXIT_CODE
fi
