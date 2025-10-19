import numpy as np
import torch
from torch import nn  # noqa: F401
import pybullet as p
from typing import Dict, Any  # noqa: F401
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# 兼容：既支持包内相对导入，也支持被单文件动态 sys.path 注入后直接 import
try:
    from .gsn_model import GainSchedulingNet  # type: ignore
except Exception:
    try:
        from gsn_model import GainSchedulingNet  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(f"无法导入 GainSchedulingNet: {e}")

class GSNController(DSLPIDControl):
    """基于神经网络的增益调度控制器：每步前向得到 P/I/D 乘数后应用到扭矩 PID 系数。
    逻辑：保持一份初始扭矩 PID 系数拷贝 (baseline)，每步乘以网络输出的倍率。
    """
    def __init__(self, drone_model, state_dim: int, device: str = 'cpu', smoothing_alpha: float = 0.2):
        super().__init__(drone_model=drone_model)
        self.device = device
        self.model = GainSchedulingNet(state_dim=state_dim).to(device)
        self.smoothing_alpha = smoothing_alpha
        self.last_gains = None  # EMA 平滑后的上一步倍率 (P,I,D)
        # 缓存初始扭矩 PID 基准增益
        self._base_P = self.P_COEFF_TOR.copy()
        self._base_I = self.I_COEFF_TOR.copy()
        self._base_D = self.D_COEFF_TOR.copy()

    def load_weights(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])

    def _build_feature_vector(self, pos_e, cur_vel, cur_quat, cur_ang_vel) -> np.ndarray:
        # roll pitch from quaternion
        roll, pitch, _ = p.getEulerFromQuaternion(cur_quat)
        wx, wy, wz = cur_ang_vel
        vx, vy, vz = cur_vel
        int_r, int_p, int_yaw = self.integral_rpy_e
        int_x, int_y, int_z = self.integral_pos_e
        return np.array([
            pos_e[0], pos_e[1], pos_e[2],
            vx, vy, vz,
            roll, pitch,
            wx, wy, wz,
            int_r, int_p, int_yaw,
            int_x, int_y, int_z,
            float(np.mean(self.P_COEFF_TOR)),
            float(np.mean(self.I_COEFF_TOR)),
            float(np.mean(self.D_COEFF_TOR))
        ], dtype=np.float32)

    def _apply_gain_multipliers(self, multipliers):
        mP, mI, mD = multipliers
        # 基于 baseline 重置后再乘，不在上次结果上继续累积
        self.P_COEFF_TOR = self._base_P * mP
        self.I_COEFF_TOR = self._base_I * mI
        self.D_COEFF_TOR = self._base_D * mD

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                       target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        # 计算位置误差用于特征
        pos_e = target_pos - cur_pos
        # 构造特征向量
        feat = self._build_feature_vector(pos_e, cur_vel, cur_quat, cur_ang_vel)
        state_tensor = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            gains = self.model(state_tensor)[0].cpu().numpy()
        # EMA 平滑
        smooth = gains if self.last_gains is None else self.smoothing_alpha * gains + (1 - self.smoothing_alpha) * self.last_gains
        self.last_gains = smooth
        # 应用倍率（在父类计算姿态控制前调节扭矩 PID 增益）
        self._apply_gain_multipliers(smooth)
        return super().computeControl(control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                                      target_pos, target_rpy, target_vel, target_rpy_rates)
