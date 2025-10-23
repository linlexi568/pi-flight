# Checkpoint 测试与可视化指南

## 📋 概述

此目录包含用于测试和可视化所有训练checkpoint性能的工具。

## 🔧 使用方法

### 1. 测试所有Checkpoint

```powershell
# 基本用法：测试所有checkpoint（自动跳过已测试的）
.venv\Scripts\python.exe analysis\test_all_checkpoints.py

# 强制重新测试所有checkpoint
.venv\Scripts\python.exe analysis\test_all_checkpoints.py --force

# 只测试前10个checkpoint
.venv\Scripts\python.exe analysis\test_all_checkpoints.py --max 10
```

**输出文件：**
- `analysis/checkpoint_test_results.csv` - 所有测试结果的CSV文件

### 2. 可视化测试结果

```powershell
# 生成所有可视化图表
.venv\Scripts\python.exe analysis\visualize_checkpoint_results.py
```

**输出图表：**（保存在 `analysis/checkpoint_figures/`）
- `00_summary.png` - 综合摘要图（推荐首先查看）
- `01_score_evolution.png` - 得分随迭代变化曲线
- `02_rules_evolution.png` - 规则数变化 + 得分vs规则数
- `03_score_distribution.png` - 得分分布直方图和箱线图
- `04_moving_average.png` - 移动平均趋势
- `05_top_checkpoints.png` - Top 10 最佳checkpoint对比

## 📊 测试配置

测试使用以下配置（与最终评估一致）：
- **轨迹预设**: `test_extreme` (5条极端轨迹)
- **聚合方式**: `harmonic` (调和平均)
- **扰动**: `stress` (4种压力扰动)
- **持续时间**: 25秒
- **奖励配置**: `pilight_boost`

具体测试的轨迹：
- coupled_surface
- zigzag3d
- lemniscate3d
- spiral_in_out
- stairs

扰动事件：
- 2.00s: 稳定风 (steady_wind)
- 7.00s: 阵风 (gusty_wind)
- 12.00s: 质量增加 (mass_up)
- 14.00s: 脉冲扰动 (pulse)

## 📁 文件说明

### `test_all_checkpoints.py`
- 自动发现所有checkpoint文件
- 批量测试并将结果写回JSON文件
- 保存汇总结果到CSV
- 支持增量测试（跳过已有verified_score的文件）

### `visualize_checkpoint_results.py`
- 读取CSV测试结果
- 生成多种可视化图表
- 分析得分趋势、规则数影响等
- 识别最佳checkpoint

### `checkpoint_test_results.csv`
测试结果CSV包含列：
- `iteration`: 迭代次数
- `num_rules`: 规则数
- `verified_score`: 测试集得分
- `train_score`: 训练集得分（如果有）
- `status`: 测试状态 (tested/cached/failed)
- `per_traj`: 各轨迹得分（JSON格式）

## 🎯 典型工作流程

```powershell
# 步骤1: 运行测试
.venv\Scripts\python.exe analysis\test_all_checkpoints.py

# 步骤2: 生成可视化
.venv\Scripts\python.exe analysis\visualize_checkpoint_results.py

# 步骤3: 查看结果
# - 打开 analysis/checkpoint_figures/00_summary.png 查看总览
# - 查看 analysis/checkpoint_test_results.csv 获取详细数据
```

## 📈 示例输出

测试完成后会显示类似摘要：

```
测试完成摘要
================================================================================
总计checkpoint数: 30
成功测试: 30
使用缓存: 25
测试失败: 0

最佳checkpoint:
  迭代: 2800
  得分: 3.247720
  规则数: 5

得分统计:
count    30.000000
mean      3.243156
std       0.005432
min       3.231045
25%       3.240123
50%       3.244567
75%       3.247890
max       3.248012
```

## 🔍 故障排查

### 问题：找不到checkpoint文件
**解决**：确保 `01_pi_flight/results/checkpoints/` 目录存在且包含文件

### 问题：测试失败
**解决**：检查虚拟环境是否激活，依赖是否完整安装

### 问题：无法生成图表
**解决**：确保已安装 matplotlib, seaborn, pandas
```powershell
pip install matplotlib seaborn pandas
```

## 💡 提示

1. 首次运行测试会比较慢（每个checkpoint约1-2分钟）
2. 后续运行会自动跳过已测试的checkpoint（除非使用`--force`）
3. 建议先查看 `00_summary.png` 获得整体概览
4. CSV文件可用Excel/Python pandas进一步分析
