#!/bin/bash
# 自动整理 01_pi_flight 目录结构
# 使用方法: bash reorganize.sh

set -e  # 出错立即退出

echo "================================================"
echo "01_pi_flight 目录整理脚本"
echo "================================================"

# 检查当前目录
if [ ! -d "01_pi_flight" ]; then
    echo "错误: 请在 pi-flight 根目录运行此脚本"
    exit 1
fi

cd 01_pi_flight

echo ""
echo "Step 0: 备份当前目录..."
BACKUP_NAME="../01_pi_flight_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
cd ..
tar -czf "$BACKUP_NAME" 01_pi_flight/
echo "✓ 备份已创建: $BACKUP_NAME"
cd 01_pi_flight

echo ""
echo "Step 1: 创建子目录..."
mkdir -p core models baselines utils deprecated
echo "✓ 子目录已创建"

echo ""
echo "Step 2: 创建 __init__.py 文件..."
touch core/__init__.py models/__init__.py baselines/__init__.py utils/__init__.py deprecated/__init__.py
echo "✓ __init__.py 文件已创建"

echo ""
echo "Step 3: 移动文件到 core/..."
for file in dsl.py program_executor.py ast_pipeline.py serialization.py; do
    if [ -f "$file" ]; then
        mv "$file" core/
        echo "  → $file → core/"
    fi
done

echo ""
echo "Step 4: 移动文件到 models/..."
for file in gnn_policy_nn_v2.py ranking_value_net.py gnn_features.py ml_param_scheduler.py; do
    if [ -f "$file" ]; then
        mv "$file" models/
        echo "  → $file → models/"
    fi
done

echo ""
echo "Step 5: 移动文件到 baselines/..."
for file in local_pid.py; do
    if [ -f "$file" ]; then
        mv "$file" baselines/
        echo "  → $file → baselines/"
    fi
done

echo ""
echo "Step 6: 移动文件到 utils/..."
for file in batch_evaluation.py reward_stepwise.py gpu_program_executor.py ultra_fast_executor.py vectorized_executor.py; do
    if [ -f "$file" ]; then
        mv "$file" utils/
        echo "  → $file → utils/"
    fi
done

echo ""
echo "Step 7: 移动文件到 deprecated/..."
for file in gnn_policy_nn.py policy_nn.py bootstrap_policy_nn.py gpu_transforms.py segmented_controller.py train_pi_flight.py test_online_system.py; do
    if [ -f "$file" ]; then
        mv "$file" deprecated/
        echo "  → $file → deprecated/"
    fi
done

echo ""
echo "================================================"
echo "✓ 整理完成!"
echo "================================================"
echo ""
echo "新的目录结构:"
tree -L 1 -d . 2>/dev/null || ls -d */
echo ""
echo "下一步:"
echo "1. 检查是否有遗漏的文件: ls *.py"
echo "2. 更新 train_online.py 的 import 语句"
echo "3. 测试: python train_online.py --help"
echo "4. 如果有问题, 恢复备份: tar -xzf $BACKUP_NAME"
echo ""
