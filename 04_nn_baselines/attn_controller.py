import collections
import numpy as np
import torch
import pybullet as p
from typing import Deque
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Relative or flat import support
try:
    from .attn_model import AttnGainNet  # type: ignore
except Exception:
    from attn_model import AttnGainNet  # type: ignore


class AttnController(DSLPIDControl):
    """Attention-based gain scheduler wrapping DSLPIDControl.

    Maintains a short window of recent feature vectors; each step predicts P/I/D
    multipliers, applies EMA smoothing, and updates torque PID gains based on
    fixed baselines (no drift accumulation).
    """

    def __init__(
        self,
        drone_model,
        state_dim: int,
        device: str = 'cpu',
        smoothing_alpha: float = 0.2,
        seq_len: int = 8,
    ) -> None:
        super().__init__(drone_model=drone_model)
        self.device = device
        self.model = AttnGainNet(state_dim=state_dim).to(device)
        self.smoothing_alpha = smoothing_alpha
        self.seq_len = max(2, int(seq_len))
        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=self.seq_len)
        self.last_gains = None
        # Baseline (fixed) torque PID gains to scale from each step
        self._base_P = self.P_COEFF_TOR.copy()
        self._base_I = self.I_COEFF_TOR.copy()
        self._base_D = self.D_COEFF_TOR.copy()

    def load_weights(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        self.model.load_state_dict(state_dict)

    def _feature(self, pos_e, cur_vel, cur_quat, cur_ang_vel) -> np.ndarray:
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
            float(int_r), float(int_p), float(int_yaw),
            float(int_x), float(int_y), float(int_z),
            float(np.mean(self.P_COEFF_TOR)),
            float(np.mean(self.I_COEFF_TOR)),
            float(np.mean(self.D_COEFF_TOR)),
        ], dtype=np.float32)

    def _apply_multipliers(self, m):
        mP, mI, mD = m
        self.P_COEFF_TOR = self._base_P * mP
        self.I_COEFF_TOR = self._base_I * mI
        self.D_COEFF_TOR = self._base_D * mD

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                       target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        pos_e = target_pos - cur_pos
        feat = self._feature(pos_e, cur_vel, cur_quat, cur_ang_vel)
        self.buffer.append(feat)
        # Pad to length seq_len for initial steps
        if len(self.buffer) < self.seq_len:
            pad = [self.buffer[0]] * (self.seq_len - len(self.buffer))
            seq = np.stack(pad + list(self.buffer), axis=0)
        else:
            seq = np.stack(self.buffer, axis=0)
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, T, F)
        with torch.no_grad():
            gains = self.model(x)[0].cpu().numpy()
        smooth = gains if self.last_gains is None else self.smoothing_alpha * gains + (1 - self.smoothing_alpha) * self.last_gains
        self.last_gains = smooth
        self._apply_multipliers(smooth)
        return super().computeControl(control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                                      target_pos, target_rpy, target_vel, target_rpy_rates)
