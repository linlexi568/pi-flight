#!/bin/bash
# Pi-Flight 统一启动脚本
# 使用方式: ./run.sh [模式] [其他参数]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_banner() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "  Pi-Flight Control Law Discovery"
    echo "  基于MCTS+GNN的程序合成系统"
    echo "=========================================="
    echo -e "${NC}"
}

show_help() {
    show_banner
    echo "使用方式: ./run.sh [模式] [选项]"
    echo ""
    echo "训练模式:"
    echo -e "  ${GREEN}quick${NC}       - 快速测试 (单agent, 100轮)"
    echo -e "  ${GREEN}pbt${NC}         - PBT完整训练 (16 agents, 5000轮)"
    echo -e "  ${GREEN}pbt-test${NC}    - PBT快速测试 (4 agents, 100轮)"
    echo -e "  ${GREEN}single${NC}      - 单agent标准训练 (自定义参数)"
    echo ""
    echo "实用工具:"
    echo -e "  ${YELLOW}check${NC}       - 检查环境依赖"
    echo -e "  ${YELLOW}clean${NC}       - 清理临时文件"
    echo -e "  ${YELLOW}logs${NC}        - 查看最近的训练日志"
    echo ""
    echo "示例:"
    echo "  ./run.sh quick                    # 快速测试"
    echo "  ./run.sh pbt --n-agents 8        # 8 agents PBT"
    echo "  ./run.sh pbt-test                # PBT快速验证"
    echo ""
}

check_env() {
    echo -e "${BLUE}检查环境...${NC}"
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}✗ Python未安装${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python: $(python --version)${NC}"
    
    # 检查必要的Python包
    python -c "import torch" 2>/dev/null || { echo -e "${RED}✗ PyTorch未安装${NC}"; exit 1; }
    echo -e "${GREEN}✓ PyTorch已安装${NC}"
    
    python -c "import torch_geometric" 2>/dev/null || { echo -e "${YELLOW}⚠ PyTorch Geometric未安装${NC}"; }
    
    # 检查GPU
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo -e "${GREEN}✓ GPU可用${NC}"
    else
        echo -e "${YELLOW}⚠ GPU不可用，将使用CPU${NC}"
    fi
    
    echo ""
}

run_quick_test() {
    echo -e "${GREEN}启动快速测试模式...${NC}"
    ./train_quick_test.sh "$@"
}

run_pbt() {
    echo -e "${GREEN}启动PBT完整训练...${NC}"
    ./train_pbt.sh "$@"
}

run_pbt_test() {
    echo -e "${GREEN}启动PBT快速测试...${NC}"
    ./pbt_test_quick.sh "$@"
}

run_single() {
    echo -e "${GREEN}启动单agent训练...${NC}"
    python 01_pi_flight/train_online.py "$@"
}

clean_temp() {
    echo -e "${YELLOW}清理临时文件...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ 清理完成${NC}"
}

show_logs() {
    echo -e "${BLUE}最近的训练日志:${NC}"
    if [ -d "logs" ]; then
        ls -lt logs/*.log | head -5
    else
        echo -e "${YELLOW}logs目录不存在${NC}"
    fi
}

# 主逻辑
MODE="${1:-help}"
shift || true  # 移除第一个参数

case "$MODE" in
    quick)
        check_env
        run_quick_test "$@"
        ;;
    pbt)
        check_env
        run_pbt "$@"
        ;;
    pbt-test)
        check_env
        run_pbt_test "$@"
        ;;
    single)
        check_env
        run_single "$@"
        ;;
    check)
        check_env
        ;;
    clean)
        clean_temp
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo -e "${RED}未知模式: $MODE${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
