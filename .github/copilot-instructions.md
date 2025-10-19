# Copilot Instructions for π-Flight

## 项目概述
π-Flight 是一个基于程序合成的无人机控制系统，使用 DSL（领域特定语言）和 MCTS（蒙特卡洛树搜索）来自动生成和优化飞行控制器。

## 项目架构与主要组件
- **核心目录**：
  - `01_pi_flight/`：核心 π-Flight 实现
    - `dsl.py`：控制器 DSL 定义
    - `train_pi_flight.py`：主训练脚本
    - `segmented_controller.py`：分段控制器实现
    - `mcts_training/`：MCTS 训练相关代码
    - `nn_training/`：神经网络训练相关代码
  - `03_CMA-ES/`：CMA-ES 优化方法基线
  - `04_decision_tree/`：决策树基线
  - `04_nn_baselines/`：神经网络基线（GSN、Attention）
  - `gym-pybullet-drones/`：基于 PyBullet 的无人机仿真环境
  - `utilities/`：通用工具函数
  - `scripts/`：自动化训练和评估脚本

## 关键开发与运行流程
- **环境配置**：
  - Python 虚拟环境：`.venv/`
  - 依赖安装：`pip install -r requirements.txt`
- **训练流程**：
  - 运行 `python 01_pi_flight/train_pi_flight.py` 进行训练
  - 支持多种轨迹：figure8, helix, circle, square 等
  - 结果保存到 `01_pi_flight/results/`
- **快速测试**：
  - VS Code 任务：`quick-train-pilight-smoke` 等
  - 参数可调：迭代次数、轨迹列表、批次大小等

## 代码风格与约定
- **控制器接口**：基于 DSL 定义的程序结构
- **仿真环境**：使用 gym-pybullet-drones 提供的 Aviary 环境
- **评估指标**：轨迹跟踪误差、稳定性等
- **日志管理**：训练日志自动保存，支持 CSV 格式

## 集成与依赖
- **外部依赖**：
  - 无人机仿真：PyBullet、gymnasium
  - 深度学习：PyTorch
  - 优化：CMA-ES
  - 可视化：matplotlib
- **仿真环境**：gym-pybullet-drones（已排除在版本控制外）

## 重要文件示例
- `01_pi_flight/train_pi_flight.py`：主训练入口
- `01_pi_flight/dsl.py`：DSL 定义和程序结构
- `01_pi_flight/segmented_controller.py`：控制器实现
- `main.py` / `main_no_gui.py`：演示脚本

## 其他注意事项
- **实验可重复性**：设置随机种子保证结果可复现
- **并行训练**：支持多进程并行评估轨迹
- **检查点保存**：定期保存最佳程序和训练状态
- **结果分析**：结果以 JSON 和 CSV 格式保存

---
如需补充或澄清具体约定，请在下方补充说明。