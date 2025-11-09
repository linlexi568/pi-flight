#!/bin/bash
################################################################################
# Pi-Flight 完整训练启动脚本
# 直接修改下方配置参数即可,无需命令行传参
################################################################################

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
# 🔧 训练参数配置 (直接改这里的数字就行)
################################################################################
CFG_ITERS=1500                 # 总迭代数
CFG_SIMS=2000                  # 每轮MCTS模拟次数 (深度搜索! 从800提升至1600)
CFG_ENVS=16384                 # 并行环境数量 (平衡初始化速度与并行度)
CFG_DURATION=10               # Episode时长(秒)
CFG_TRAJ="hover"              # 任务: hover/tracking
CFG_EVAL_REPLICAS=5           # 每程序评估次数 (从4提升至5, 更稳定)
CFG_UPDATE_FREQ=8             # 每N轮更新GNN (从10降至8, 更频繁学习)
CFG_TRAIN_STEPS=12            # 每次更新训练步数 (从8提升至12, 更充分训练)
CFG_BATCH_SIZE=768            # 批大小 (从512提升至768, 更大批量)
CFG_MIN_STEPS_FRAC=0.3        # 最短步数比例
CFG_LR=1e-3                   # 学习率
CFG_CHECKPOINT_FREQ=40       # Checkpoint保存频率 (从50降至40, 更频繁保存)
CFG_REWARD_REDUCTION="mean"   # 奖励聚合: mean/max/last

# 🎯 性能优化说明:
#   🔥 激进配置 (在8GB显存限制下最大化性能):
#   
#   1️⃣ MCTS深度: 800 → 1600 (2×搜索深度)
#      - 每轮探索更多程序空间
#      - 发现更优策略的概率翻倍
#   
#   2️⃣ 环境数: 8192 → 16384 (2×并行度)
#      - 初始化时间: <0.3秒 (可接受)
#      - 满足需求: 1600程序×5 replicas = 8000 < 16384 ✅
#      - 显存安全: 16384环境约占用5-6GB (RTX 4060 8GB安全范围)
#   
#   3️⃣ 评估稳定性: 4 → 5 replicas
#      - 每个程序评估5次取平均, 减少噪声
#   
#   4️⃣ 学习频率: 每10轮 → 每8轮更新GNN
#      - 更快响应搜索发现的优质程序
#   
#   5️⃣ 训练强度: 8步 → 12步/次
#      - 每次更新更充分拟合数据
#   
#   6️⃣ 批大小: 512 → 768
#      - 更大批量, 梯度估计更准确
#   
#   🚀 超高性能优化 (use-fast-path):
#   - 环境池持久化复用 (避免反复初始化)
#   - Numba JIT编译 (机器码执行, 无GIL, 多核并行)
#   - 程序预编译缓存 (1600相同程序→1个缓存)
#   
#   📈 预估效果:
#   - 单迭代: ~22秒 (1600程序比800程序慢约50%)
#   - 2000迭代: 约12小时 (可过夜训练)
#   - 但搜索质量提升2×, 收敛更快, 可能1000轮就达到2000轮效果!
################################################################################

# 创建目录
mkdir -p logs results 01_pi_flight/results

# 生成时间戳文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_training_${TIMESTAMP}.log"
RESULT_FILE="01_pi_flight/results/phase1_full_training_${TIMESTAMP}.json"

# 显示配置
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║             Pi-Flight 完整训练 - Phase 1 Hover                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📋 训练配置:${NC}"
echo -e "  模式: AlphaZero (MCTS + GNN)"
echo -e "  任务: ${CFG_TRAJ}"
echo -e "  迭代: ${CFG_ITERS} 轮"
echo -e "  MCTS: ${CFG_SIMS} 次模拟/轮"
echo -e "  环境: ${CFG_ENVS} 个并行"
echo -e "  时长: ${CFG_DURATION} 秒/episode"
echo -e "  评估: ${CFG_EVAL_REPLICAS} 次/程序"
echo -e "  学习率: ${CFG_LR}"
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
# export DEBUG_ENV_POOL=1  # 取消注释可查看环境池复用日志

"$VENV_PYTHON" 01_pi_flight/train_online.py \
    --use-gnn \
    --use-fast-path \
    --traj "${CFG_TRAJ}" \
    --duration "${CFG_DURATION}" \
    --total-iters "${CFG_ITERS}" \
    --mcts-simulations "${CFG_SIMS}" \
    --real-sim-frac 1.0 \
    --update-freq "${CFG_UPDATE_FREQ}" \
    --train-steps-per-update "${CFG_TRAIN_STEPS}" \
    --batch-size "${CFG_BATCH_SIZE}" \
    --isaac-num-envs "${CFG_ENVS}" \
    --eval-replicas-per-program "${CFG_EVAL_REPLICAS}" \
    --min-steps-frac "${CFG_MIN_STEPS_FRAC}" \
    --reward-reduction "${CFG_REWARD_REDUCTION}" \
    --learning-rate "${CFG_LR}" \
    --checkpoint-freq "${CFG_CHECKPOINT_FREQ}" \
    --save-path "$RESULT_FILE" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# 显示结果
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 训练完成!${NC}"
    echo -e "${BLUE}📊 结果: $RESULT_FILE${NC}"
    echo -e "${BLUE}📄 日志: $LOG_FILE${NC}"
else
    echo -e "${RED}❌ 训练失败 (退出码: $EXIT_CODE)${NC}"
    echo -e "${YELLOW}查看日志: $LOG_FILE${NC}"
    exit $EXIT_CODE
fi
