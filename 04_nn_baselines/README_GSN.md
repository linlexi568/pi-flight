# Gain Scheduling Network (GSN)

该子模块提供一个神经网络基线，用于与 π-Light 分段 PID 进行可解释性与性能对比。

## 组件
- `gsn_model.py`: `GainSchedulingNet` 输出 P/I/D 增益乘数 (安全区间映射)
- `gsn_controller.py`: 封装控制器, 在调用父类 PID 计算前动态调整扭矩 PID 增益
- `dataset_builder.py`: 运行 π-Light 控制器采集 (state -> 实际使用的增益倍率) 数据集
- `train_gsn.py`: 监督训练脚本, 读取采集的 npz 数据, 拟合增益调度映射

## 数据格式
`results/gsn_dataset.npz` 包含:
- `states`: (N, F)
- `gains`:  (N, 3) -> [mP, mI, mD]
- `meta`:   JSON字符串 (列说明 / state_dim)

## 训练示例
```bash
python -m nn_baselines.dataset_builder -- (直接运行默认示例或自行修改)
python -m nn_baselines.train_gsn --data results/gsn_dataset.npz --epochs 20 --bs 256 --lr 3e-4
```
输出: `checkpoints/gsn_best.pt`

## 在主程序中使用 (示意)
```python
from nn_baselines.gsn_controller import GSNController
from gym_pybullet_drones.utils.enums import DroneModel
import torch

controller = GSNController(DroneModel("cf2x"), state_dim=20)
ckpt = torch.load('checkpoints/gsn_best.pt', map_location='cpu')
controller.model.load_state_dict(ckpt['model'])
# 然后将 controller 传给 SimulationTester
```

## 后续可扩展
- Residual Policy / PPO 端到端基线
- 轨迹多样化采集 + 数据混合
- TensorBoard 训练可视化
- Ablation: 去掉积分项 / 去掉姿态特征的影响分析
