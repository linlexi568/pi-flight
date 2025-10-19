"""Decision Tree Controller for PID gain scheduling."""

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

try:
    from .dt_model import DTGainScheduler
except ImportError:
    from dt_model import DTGainScheduler


class DTController(DSLPIDControl):
    """Decision Tree based PID gain scheduling controller.
    
    Similar to GSN but uses interpretable decision trees instead of neural networks.
    """
    
    def __init__(self, drone_model, state_dim: int = 20, smoothing_alpha: float = 0.2):
        """
        Args:
            drone_model: Drone model enum
            state_dim: Dimension of state features
            smoothing_alpha: EMA smoothing factor for gain transitions
        """
        super().__init__(drone_model=drone_model)
        self.state_dim = state_dim
        self.smoothing_alpha = smoothing_alpha
        self.model = DTGainScheduler(state_dim=state_dim)
        self.last_gains = None
        
        # Cache baseline torque PID gains
        self._base_P = self.P_COEFF_TOR.copy()
        self._base_I = self.I_COEFF_TOR.copy()
        self._base_D = self.D_COEFF_TOR.copy()
    
    def load_model(self, path: str):
        """Load trained decision tree model."""
        self.model.load(path)
    
    def _build_feature_vector(self, pos_e, cur_vel, cur_quat, cur_ang_vel) -> np.ndarray:
        """Build state feature vector (same as GSN for fair comparison)."""
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
            float(np.mean(self._base_P)),
            float(np.mean(self._base_I)),
            float(np.mean(self._base_D))
        ], dtype=np.float32)
    
    def _apply_gain_multipliers(self, multipliers):
        """Apply gain multipliers to baseline PID coefficients."""
        mP, mI, mD = multipliers
        self.P_COEFF_TOR = self._base_P * mP
        self.I_COEFF_TOR = self._base_I * mI
        self.D_COEFF_TOR = self._base_D * mD
    
    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                       target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), 
                       target_rpy_rates=np.zeros(3)):
        """Compute control with decision tree gain scheduling."""
        # Compute position error
        pos_e = target_pos - cur_pos
        
        # Build feature vector
        feat = self._build_feature_vector(pos_e, cur_vel, cur_quat, cur_ang_vel)
        
        # Predict gains using decision tree
        gains = self.model.predict(feat)  # (3,) [P, I, D]
        
        # EMA smoothing
        if self.last_gains is None:
            smooth = gains
        else:
            smooth = self.smoothing_alpha * gains + (1 - self.smoothing_alpha) * self.last_gains
        self.last_gains = smooth
        
        # Apply multipliers
        self._apply_gain_multipliers(smooth)
        
        # Call parent PID computation
        return super().computeControl(control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                                      target_pos, target_rpy, target_vel, target_rpy_rates)
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self.last_gains = None


if __name__ == '__main__':
    from gym_pybullet_drones.utils.enums import DroneModel
    
    controller = DTController(DroneModel.CF2X, state_dim=20)
    print("DTController initialized successfully")
    print(f"State dim: {controller.state_dim}")
    print(f"Model trained: {controller.model.is_trained}")
