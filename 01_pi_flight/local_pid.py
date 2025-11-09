from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# 一个轻量的本地 PID 控制器，提供与 DSLPIDControl 近似的接口
# 目标：满足 segmented_controller 的基类需求（属性与方法签名），便于在 Isaac Gym 下运行


@dataclass
class LocalDroneModel:
    name: str = "cf2x"


class SimplePIDControl:
    def __init__(self, drone_model: LocalDroneModel | str = "cf2x"):
        # 记录机型（当前仅占位，不分机型）
        self.drone_model = LocalDroneModel(str(drone_model)) if not isinstance(drone_model, LocalDroneModel) else drone_model
        # 角度环 PID 系数（力矩通道）——作为“默认增益”，分段控制会在此基础上做乘法
        self.P_COEFF_TOR = np.array([2.5, 2.5, 1.0], dtype=float)
        self.I_COEFF_TOR = np.array([0.02, 0.02, 0.02], dtype=float)
        self.D_COEFF_TOR = np.array([0.15, 0.15, 0.05], dtype=float)
        # 位置环（外环）简化参数，用于从位置误差生成期望姿态
        self.K_P_POS = np.array([0.8, 0.8, 1.5], dtype=float)
        self.K_D_POS = np.array([0.2, 0.2, 0.6], dtype=float)
        # 积分记忆
        self.integral_rpy_e = np.zeros(3, dtype=float)
        self.integral_pos_e = np.zeros(3, dtype=float)
        # 物理与混控常数（简化）
        self.mass = 0.027  # kg
        self.g = 9.81
        # 推力/扭矩到电机 thrust 的 mixing 常数（极简近似）
        self.arm = 0.046  # m（CF 臂长近似）
        self.kf = 3.16e-10  # N/(rad/s)^2
        self.km = 1.5e-12   # Nm/(rad/s)^2（估计值）
        # 电机 rpm 限幅
        self.rpm_min = 0.0
        self.rpm_max = 21702.0

    def _euler_from_quat(self, q):
        # 四元数 -> 欧拉 XYZ（roll, pitch, yaw）
        x, y, z, w = q
        # 来自标准转换
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=float)

    def _attitude_pid(self, rpy_err, ang_vel, dt):
        # 基于姿态误差与角速度的 PID 输出期望力矩（简化）
        self.integral_rpy_e += rpy_err * dt
        p_term = self.P_COEFF_TOR * rpy_err
        i_term = self.I_COEFF_TOR * self.integral_rpy_e
        d_term = self.D_COEFF_TOR * (-ang_vel)
        torques = p_term + i_term + d_term
        return torques

    def _pos_outer_loop(self, pos_err, vel, dt):
        # 外环：从位置误差生成期望姿态（roll, pitch），及总推力
        # 水平面：期望 roll/pitch 与位置误差成比例；Z 轴：高度 PID -> 总推力
        # 注意：真实系统需坐标变换，这里做简化
        desired_roll = + self.K_P_POS[1] * pos_err[1] - self.K_D_POS[1] * vel[1]
        desired_pitch = - self.K_P_POS[0] * pos_err[0] - self.K_D_POS[0] * vel[0]
        desired_yaw = 0.0
        # 高度：期望总推力 = m(g + z环输出)
        z_u = self.K_P_POS[2] * pos_err[2] - self.K_D_POS[2] * vel[2]
        total_thrust = self.mass * (self.g + z_u)
        return np.array([desired_roll, desired_pitch, desired_yaw], dtype=float), float(np.clip(total_thrust, 0.0, 2.5*self.mass*self.g))

    def _mix_to_motors(self, total_thrust, torques):
        # 极简四旋翼 X 架混控：
        # f1 = T/4 - τx/(2l) - τy/(2l) - τz/(4km/kf)
        # f2 = T/4 - τx/(2l) + τy/(2l) + τz/(4km/kf)
        # f3 = T/4 + τx/(2l) + τy/(2l) - τz/(4km/kf)
        # f4 = T/4 + τx/(2l) - τy/(2l) + τz/(4km/kf)
        l = self.arm
        tx, ty, tz = torques
        c = self.km / self.kf if self.kf > 0 else 1e-6
        t4 = total_thrust / 4.0
        f1 = t4 - tx/(2*l) - ty/(2*l) - tz/(4*c)
        f2 = t4 - tx/(2*l) + ty/(2*l) + tz/(4*c)
        f3 = t4 + tx/(2*l) + ty/(2*l) - tz/(4*c)
        f4 = t4 + tx/(2*l) - ty/(2*l) + tz/(4*c)
        thrusts = np.clip(np.array([f1, f2, f3, f4], dtype=float), 0.0, None)
        # thrust -> omega -> rpm
        omega = np.sqrt(thrusts / max(self.kf, 1e-9))
        rpm = omega * 60.0 / (2*np.pi)
        rpm = np.clip(rpm, self.rpm_min, self.rpm_max)
        return rpm

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                       target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        # 位置外环 -> 期望姿态 & 总推力
        pos_err = target_pos - cur_pos
        desired_rpy, total_thrust = self._pos_outer_loop(pos_err, cur_vel, control_timestep)
        # 姿态误差
        rpy = self._euler_from_quat(cur_quat)
        rpy_err = desired_rpy - rpy
        # 姿态内环 -> 力矩
        torques = self._attitude_pid(rpy_err, cur_ang_vel, control_timestep)
        # 混控到电机
        rpm = self._mix_to_motors(total_thrust, torques)
        return rpm, pos_err, rpy_err

    def computeControlFromState(self, control_timestep, state, target_pos):
    # 兼容旧测试接口：state 向量需包含 [x,y,z,vx,vy,vz, roll, pitch, yaw, wx, wy, wz] 或类似顺序
    # 这里假设 state 的排列与常见四旋翼状态约定一致：
        # 0:x 1:y 2:z 3:vx 4:vy 5:vz 6:r 7:p 8:yaw 9:wx 10:wy 11:wz ...
        cur_pos = np.array(state[0:3], dtype=float)
        cur_vel = np.array(state[3:6], dtype=float)
        rpy = np.array(state[6:9], dtype=float)
        ang = np.array(state[9:12], dtype=float)
        # 从 rpy 重建四元数（近似，yaw-only 误差容忍）
        cr = np.cos(rpy[0]*0.5); sr=np.sin(rpy[0]*0.5)
        cp = np.cos(rpy[1]*0.5); sp=np.sin(rpy[1]*0.5)
        cy = np.cos(rpy[2]*0.5); sy=np.sin(rpy[2]*0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        cur_quat = np.array([qx,qy,qz,qw], dtype=float)
        rpm, pos_err, rpy_err = self.computeControl(control_timestep, cur_pos, cur_quat, cur_vel, ang, target_pos)
        return rpm, pos_err, rpy_err
