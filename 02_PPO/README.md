# PPO Baseline Implementation

这个目录包含使用Stable-Baselines3实现的PPO baseline,用于与我们的程序合成方法对比。

## 文件结构

```
02_PPO/
├── README.md              # 本文件
├── baseline_ppo.py        # PPO训练和评估主文件
├── requirements.txt       # Python依赖
├── train.sh              # 训练脚本
└── results/              # 实验结果目录
    ├── circle/
    ├── figure8/
    └── hover_wind/
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练PPO

```bash
# Circle任务
python baseline_ppo.py --mode train --task circle --timesteps 1000000 --n-envs 8

# Figure-8任务
python baseline_ppo.py --mode train --task figure8 --timesteps 1000000 --n-envs 8

# Hover with Wind任务
python baseline_ppo.py --mode train --task hover_wind --timesteps 1000000 --n-envs 8
```

或使用脚本:
```bash
bash train.sh
```

### 3. 评估训练好的模型

```bash
python baseline_ppo.py --mode eval --task circle --n-eval 100
```

## 超参数设置

我们使用Stable-Baselines3的标准PPO超参数:

- Learning rate: 3e-4
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01

## 实验结果

### Circle Flight

| Metric | Value |
|--------|-------|
| Mean Reward | TODO |
| Std Reward | TODO |
| Training Time | TODO |
| Model Size | ~180K params |

### Figure-8

| Metric | Value |
|--------|-------|
| Mean Reward | TODO |
| Std Reward | TODO |

### Hover with Wind

| Metric | Value |
|--------|-------|
| Mean Reward | TODO |
| Std Reward | TODO |

## 与程序合成方法对比

| Method | Circle | Figure-8 | Hover-Wind | Interpretable |
|--------|--------|----------|------------|---------------|
| PPO (Ours) | TODO | TODO | TODO | ❌ |
| Program Synthesis | TODO | TODO | TODO | ✅ |

## Notes

- PPO使用黑盒神经网络 (~180K参数)
- 不可解释
- 需要大量训练数据
- 部署需要完整的NN推理
