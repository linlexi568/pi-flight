# PPO Baseline Implementation

这个目录包含使用 Stable-Baselines3 实现的 PPO baseline，用于与 π-Flight 程序合成方法进行对比。

## 核心特点

- **纯 Python 配置**: 所有参数硬编码在 `baseline_ppo.py` 的 `_FixedConfig` 类中，不接受命令行参数。
- **Isaac Gym 集成**: 使用自定义 Wrapper 直接对接 Isaac Gym 的并行环境 (默认 512 个环境)。
- **Reward Profile**: 支持与 π-Flight 相同的四种奖励配置 (`safety_first`, `tracking_first`, `balanced`, `robustness_stability`)。

## 文件结构

```
02_PPO/
├── baseline_ppo.py        # 核心脚本：包含配置、环境Wrapper、训练/评估逻辑
├── README.md              # 说明文档
└── results/               # 结果保存目录 (自动生成)
    └── {task}_{profile}/  # 例如 figure8_balanced/
        ├── tensorboard/   # Tensorboard 日志
        ├── checkpoints/   # 中间模型权重
        └── final_model.zip # 最终模型
```

## 使用说明

### 1. 配置参数

打开 `baseline_ppo.py`，找到 `_FixedConfig` 类进行修改：

```python
class _FixedConfig:
    # --- 实验设置 ---
    mode = 'train'          # 'train' (训练) 或 'eval' (评估)
    
    # --- 环境设置 ---
    task = 'figure8'        # 任务: 'hover', 'figure8', 'circle', 'helix'
    duration = 10.0         # 单次 episode 时长 (秒)
    isaac_num_envs = 512    # 并行环境数量 (建议保持 512 以获得高吞吐量)
    
    # --- 奖励设置 ---
    # 可选: 'safety_first', 'tracking_first', 'balanced', 'robustness_stability'
    reward_profile = 'balanced'
    
    # --- 训练参数 ---
    total_timesteps = 1_000_000  # 总训练步数
    # ... 其他 PPO 超参数
```

### 2. 运行脚本

直接运行 Python 脚本（确保在激活的虚拟环境中）：

```bash
# 确保已激活环境 (例如 source ../.venv/bin/activate)
python baseline_ppo.py
```

### 3. 查看结果

训练过程中会输出进度条。训练完成后，模型保存在 `02_PPO/results/` 下。

启动 Tensorboard 查看曲线：

```bash
tensorboard --logdir 02_PPO/results
```

## 注意事项

- **环境兼容性**: 脚本内部处理了 `gym` (Isaac Gym) 和 `gymnasium` (Stable-Baselines3) 的兼容性问题。
- **性能**: 使用 512 个并行环境，100万步通常只需要几分钟（取决于 GPU）。
- **依赖**: 需要安装 `stable-baselines3`, `shimmy`, `tqdm`, `rich` 等库 (已包含在项目环境中)。
