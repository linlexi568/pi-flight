# analysis 目录

这里汇总了对 π-Flight 最优程序的各类分析与对比脚本/文档，并统一输出图表与摘要。

## 📂 目录说明

### 🎯 Checkpoint 测试与分析（新增）
- **`test_all_checkpoints.py`** - 批量测试所有训练checkpoint性能
- **`visualize_checkpoint_results.py`** - 可视化checkpoint测试结果，生成多种分析图表
- **`run_checkpoint_analysis.ps1`** - 一键测试+可视化脚本（推荐使用）
- **`README_CHECKPOINT_TESTING.md`** - Checkpoint测试详细使用指南
- **`checkpoint_test_results.csv`** - 测试结果数据（自动生成）
- **`checkpoint_figures/`** - 生成的图表目录（自动创建）

### 📊 综合分析工具
- **`collect_comprehensive_data.py`** - 收集 3×3 组合（轨迹集 × 扰动等级）的真实评测数据，生成 CSV
- **`visualize_comprehensive_comparison_final.py`** - 基于 CSV 生成 3 张对比可视化图表
- **`analyze_comparison_depth.py`** - 对 9 组对比做统计分析（平均优势、胜场数、敏感性等）
- **`analyze_best_program_deep.py`** - 解析 `01_pi_flight/results/best_program.json`，输出规则数与关键信息摘要
- **`control_theory_analysis.py`** - 控制理论分析（Lyapunov、传递函数、Bode、敏感度/互补敏感度、时域）

### 📝 文档与总结
- **`analyze_interpretability.md`** - 可解释性对比（π-Flight vs CMA-ES vs 神经网络）
- **`summary_advantages.md`** - π-Flight 优势总览（性能、稳健、可维护、安全）

## 运行先决条件
- 已安装项目依赖（建议在 .venv 虚拟环境中）：`pip install -r requirements.txt`
- 评测依赖：需要能无头运行 gym-pybullet-drones（本仓库已包含源码）。

## 🚀 快速开始（Windows PowerShell）

### Checkpoint 分析（推荐先运行）

```powershell
# 方式1: 使用一键脚本（交互式菜单）
.\analysis\run_checkpoint_analysis.ps1

# 方式2: 手动运行完整流程
.\.venv\Scripts\python.exe analysis\test_all_checkpoints.py
.\.venv\Scripts\python.exe analysis\visualize_checkpoint_results.py
explorer analysis\checkpoint_figures  # 查看生成的图表
```

### 综合分析

1) 控制理论分析（快速、无仿真）：

```powershell
& .\.venv\Scripts\python.exe analysis\control_theory_analysis.py
```

2) 收集 3×3 评测数据（耗时较长，18 次评测）：

```powershell
& .\.venv\Scripts\python.exe analysis\collect_comprehensive_data.py
```

3) 基于收集到的数据生成图表：

```powershell
& .\.venv\Scripts\python.exe analysis\visualize_comprehensive_comparison_final.py
```

4) 统计分析与摘要：

```powershell
& .\.venv\Scripts\python.exe analysis\analyze_comparison_depth.py
& .\.venv\Scripts\python.exe analysis\analyze_best_program_deep.py
```

## 📁 输出位置

### Checkpoint 分析输出
- 图表：`analysis/checkpoint_figures/` (6张PNG图表)
- 数据：`analysis/checkpoint_test_results.csv`
- 详细指南：`analysis/README_CHECKPOINT_TESTING.md`

### 综合分析输出
- 图表：`results/figures/` 与 `results/control_theory_analysis/`
- 数据：`results/summaries/comprehensive_comparison_results_collected.csv`
- 文本摘要：`results/summaries/` 下的若干 `.md` 文件

## 💡 使用建议

1. **首次运行**：使用 `run_checkpoint_analysis.ps1` 进行checkpoint分析，了解训练过程
2. **快速查看**：打开 `checkpoint_figures/00_summary.png` 查看训练总览
3. **深度分析**：运行综合分析脚本，获取详细对比数据
4. **文档参考**：查看 `README_CHECKPOINT_TESTING.md` 了解详细配置

> 提示：如果某些数据文件暂不存在，脚本会给出友好的提示并指导如何生成。