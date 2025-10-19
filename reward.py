# reward.py
import numpy as np
import pandas as pd
import math
from typing import Optional, List, Dict

class RewardCalculator:
    """
    一个模块化的奖励计算器，用于评估无人机悬停控制器的性能。
    它接收一个pandas DataFrame格式的飞行日志，并根据多个维度计算一个综合奖励分数。
    """

    def __init__(self,
                 weights: dict,
                 target_pos: np.ndarray = np.array([0, 0, 0.5]),
                 settling_time_threshold: float = 0.02,  # 2cm
                 dt: float = 1/48.0,  # 控制周期
                 trajectory_df: Optional[pd.DataFrame] = None
                 ):
        """
        初始化奖励计算器。

        Args:
            weights (dict): 一个包含各项奖励权重的字典，此参数为必需。
                            例如: {'position_rmse': 5.0, 'settling_time': 2.0, ...}
            target_pos (np.ndarray): 悬停的目标位置 [x, y, z]。
            settling_time_threshold (float): 计算整定时间时使用的位置误差阈值 (米)。
            dt (float): 日志记录的时间间隔 (秒)。
            trajectory_df (pd.DataFrame, optional): 包含目标轨迹的DataFrame。
        """
        self.target_pos = target_pos
        self.settling_time_threshold = settling_time_threshold
        self.dt = dt
        # 兼容：补齐新权重键
        default_extra = {
            'gain_stability': 0.0,   # 默认不启用，调用方可显式赋值
            'saturation': 0.0,
            'peak_error': 0.0
        }
        for k, v in default_extra.items():
            if k not in weights:
                weights[k] = v
        self.weights = weights
        self.trajectory_df = trajectory_df
            
    def _calculate_position_rmse(self, log_df: pd.DataFrame) -> float:
        """计算位置均方根误差。如果提供了轨迹，则计算跟踪误差。"""
        if self.trajectory_df is not None:
            # 将日志df的时间戳设置为索引，以便与轨迹df对齐
            log_df_indexed = log_df.set_index('timestamp')
            
            # 使用 asof 合并，为每个实际状态点找到最近的先前目标点
            merged_df = pd.merge_asof(
                log_df_indexed.sort_index(),
                self.trajectory_df.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest' # 找到最近的目标点
            )
            
            actual_pos = merged_df[['x', 'y', 'z']].values
            target_pos = merged_df[['target_x', 'target_y', 'target_z']].values
            
            # 过滤掉合并失败的行
            valid_rows = ~np.isnan(target_pos).any(axis=1)
            if not np.any(valid_rows):
                print("[警告] 无法将日志与轨迹对齐。")
                return 1e9

            errors = np.linalg.norm(actual_pos[valid_rows] - target_pos[valid_rows], axis=1)
        else:
            # 原始的悬停误差计算
            pos = log_df[['x', 'y', 'z']].values
            errors = np.linalg.norm(pos - self.target_pos, axis=1)

        rmse = np.sqrt(np.mean(errors**2))
        return rmse + 1e-9

    def _calculate_avg_settling_time(self, log_df: pd.DataFrame, disturbance_times: list) -> float:
        """如果正在跟踪轨迹，则此指标无意义，返回一个小的惩罚值。"""
        if self.trajectory_df is not None:
            return 1.0 # 返回一个固定的惩罚值，因为稳定时间不适用于轨迹跟踪

        if not disturbance_times:
            return 1e-9
        pos = log_df[['x', 'y', 'z']].values
        errors = np.linalg.norm(pos - self.target_pos, axis=1)
        total_settling_time = 0
        for t_disturbance in disturbance_times:
            start_index = int(t_disturbance / self.dt)
            settled = False
            for i in range(start_index, len(errors)):
                if errors[i] <= self.settling_time_threshold:
                    settling_time = (i - start_index) * self.dt
                    total_settling_time += settling_time
                    settled = True
                    break
            if not settled:
                total_settling_time += (len(errors) - start_index) * self.dt
        return (total_settling_time / len(disturbance_times)) + 1e-9

    def _calculate_control_effort(self, log_df: pd.DataFrame) -> float:
        rpms = log_df[['rpm0', 'rpm1', 'rpm2', 'rpm3']].values
        rpm_diffs = np.abs(np.diff(rpms, axis=0))
        total_variation = np.sum(rpm_diffs)
        normalized_variation = total_variation / (len(log_df) * self.dt)
        return normalized_variation + 1e-9

    def _calculate_smoothness_jerk(self, log_df: pd.DataFrame) -> float:
        vels = log_df[['vx', 'vy', 'vz']].values
        if len(vels) < 3: return 1e9
        accels = np.diff(vels, axis=0) / self.dt
        jerks = np.diff(accels, axis=0) / self.dt
        jerk_magnitude_rms = np.sqrt(np.mean(np.sum(jerks**2, axis=1)))
        return jerk_magnitude_rms + 1e-9

    def compute_reward(self,
                       log_df: pd.DataFrame,
                       disturbance_times: Optional[list] = None,
                       gain_history: Optional[List[Dict[str, float]]] = None,
                       saturation_events: Optional[int] = None,
                       peak_error: Optional[float] = None,
                       max_motor_thrust: Optional[float] = None,
                       verbose: bool = True
                       ) -> float:
        """计算总奖励。

        可选新增参数:
            gain_history: [{'P':..,'I':..,'D':..}, ...]
            saturation_events: 电机接近饱和(>=98%) 的时间步计数
            peak_error: 运行期间出现的最大位置误差 (米)
        """
        # ... 保持原有主要结构，新增扩展项 ...
        if disturbance_times is None:
            disturbance_times = []
        score_rmse = self._calculate_position_rmse(log_df)
        score_settling_time = self._calculate_avg_settling_time(log_df, disturbance_times)
        score_effort = self._calculate_control_effort(log_df)
        score_jerk = self._calculate_smoothness_jerk(log_df)
        # ---------------- 新增：增益稳定性 ----------------
        gain_stability_cost = 0.0
        if gain_history and len(gain_history) > 1 and self.weights.get('gain_stability', 0) > 0:
            df_g = pd.DataFrame(gain_history)
            ref = df_g.iloc[0].replace(0, np.nan)
            rel = df_g / ref
            # 使用标准差（也可用方差）
            gain_stability_cost = rel.std().mean()

        # ---------------- 新增：电机饱和比例 ----------------
        saturation_ratio = 0.0
        if saturation_events is not None and len(log_df) > 0 and self.weights.get('saturation', 0) > 0:
            saturation_ratio = saturation_events / len(log_df)

        # ---------------- 新增：峰值误差 ----------------
        if peak_error is None:
            # 若未显式提供则用 RMSE 上界近似（保守）。也可再计算一次最大值：
            if self.trajectory_df is not None:
                # 与轨迹对齐后最大误差（复用 _calculate_position_rmse 中逻辑较重，此处简化）
                # 简易方式：计算 log_df 与 target_pos 或假定 rmse ~ peak/2 => peak≈2*rmse
                peak_error_est = 2 * score_rmse
            else:
                pos = log_df[['x', 'y', 'z']].values
                errors = np.linalg.norm(pos - self.target_pos, axis=1)
                peak_error_est = np.max(errors) if len(errors) else score_rmse
            peak_error = peak_error_est
        peak_error_cost = float(peak_error if peak_error is not None else 0.0)

        # 指数衰减系数（可后续调参）
        k_rmse = 1.7
        k_settling = 0.13
        k_effort = 0.0000013
        k_jerk = 0.0057
        k_gain = 1.2       # 增益稳定性
        k_sat = 4.0        # 饱和
        k_peak = 1.2       # 峰值误差

        reward = (
            self.weights['position_rmse'] * math.exp(-k_rmse * score_rmse) +
            self.weights['settling_time'] * math.exp(-k_settling * score_settling_time) +
            self.weights['control_effort'] * math.exp(-k_effort * score_effort) +
            self.weights['smoothness_jerk'] * math.exp(-k_jerk * score_jerk) +
            self.weights['gain_stability'] * math.exp(-k_gain * gain_stability_cost) +
            self.weights['saturation'] * math.exp(-k_sat * saturation_ratio) +
            self.weights['peak_error'] * math.exp(-k_peak * peak_error_cost)
        )

        if verbose:
            print(
                f"综合奖励: {reward:.4f} | rmse={score_rmse:.3f} | settle={score_settling_time:.3f} | effort={score_effort:.2e} | jerk={score_jerk:.3f} | "
                f"gain_std={gain_stability_cost:.3f} | sat={saturation_ratio:.3f} | peak_err={peak_error_cost:.3f}"
            )
        return reward







