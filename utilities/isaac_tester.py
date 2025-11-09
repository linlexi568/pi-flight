import time
import os, importlib.util
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from scipy.spatial.transform import Rotation

# 动态加载 Isaac Gym 环境类，避免包名以数字开头导致的导入问题
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_ROOT, '01_pi_flight', 'envs', 'isaac_gym_drone_env.py')
_ENV_MOD_NAME = 'piflight_env.isaac_gym_drone_env'
spec = importlib.util.spec_from_file_location(_ENV_MOD_NAME, _ENV_FILE)
if spec and spec.loader:
    _mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[_ENV_MOD_NAME] = _mod
    spec.loader.exec_module(_mod)  # type: ignore
    IsaacGymDroneEnv = getattr(_mod, 'IsaacGymDroneEnv')
else:
    raise ImportError('Failed to load IsaacGymDroneEnv')

from reward import RewardCalculator

# 轻量 CPU 后备环境（当 Isaac Gym 不可用时用于冒烟测试）
class SimpleCPUHoverEnv:
    def __init__(self, num_envs: int = 1, control_freq_hz: int = 48, duration_sec: int = 4):
        import torch
        self.num_envs = num_envs
        self.device = torch.device('cpu')
        self.control_freq_hz = control_freq_hz
        self.dt = 1.0 / float(control_freq_hz)
        self.duration_sec = duration_sec
        # 状态张量，形状尽量贴近 Isaac 接口
        self.pos = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.pos[:, 2] = 1.0  # 初始高度
        self.quat = torch.zeros((num_envs, 4), dtype=torch.float32)
        self.quat[:, 3] = 1.0  # 单位四元数
        self.lin_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        # 简化动力学参数（与 Crazyflie 接近）
        self.mass = 0.027
        self.g = 9.81
        self.KF = 3.16e-10

    def get_states_batch(self) -> Dict[str, Any]:
        return {
            'pos': self.pos.clone(),
            'vel': self.lin_vel.clone(),
            'quat': self.quat.clone(),
            'omega': self.ang_vel.clone(),
        }

    def reset(self):
        import torch
        self.pos[:] = 0.0
        self.pos[:, 2] = 1.0
        self.lin_vel[:] = 0.0
        self.ang_vel[:] = 0.0
        return self.get_states_batch()

    def step(self, actions):
        # actions: [N,4] RPM
        import torch
        if actions.shape[-1] != 4:
            raise ValueError('SimpleCPUHoverEnv 仅支持 [N,4] RPM 动作')
        # RPM -> thrust (N)
        omega = actions.float() * (2.0 * 3.1415926535 / 60.0)
        T = self.KF * (omega ** 2)  # [N,4]
        Fz = torch.sum(T, dim=1)  # 合力（机体向上）
        # 只做竖直方向简化积分
        acc_z = (Fz / self.mass) - self.g
        self.lin_vel[:, 2] += acc_z * self.dt
        self.pos[:, 2] += self.lin_vel[:, 2] * self.dt
        # 地面碰撞近似
        below = self.pos[:, 2] < 0.0
        if torch.any(below):
            self.pos[below, 2] = 0.0
            self.lin_vel[below, 2] = 0.0
        # 其他返回值按 Isaac Gym 接口占位
        obs = {
            'position': self.pos.numpy(),
            'velocity': self.lin_vel.numpy(),
            'orientation': self.quat.numpy(),
            'angular_velocity': self.ang_vel.numpy(),
        }
        rewards = torch.zeros((self.num_envs,), dtype=torch.float32)
        dones = torch.zeros((self.num_envs,), dtype=torch.bool)
        return obs, rewards, dones, {}

# 单例环境，避免反复创建/销毁导致底层 Foundation 冲突
_ENV_SINGLETON = None  # type: ignore

class SimulationTester:
    """
    使用 Isaac Gym 环境的单环境测试器，提供与原 test.py 近似的接口，
    以便 verify_program/main 等脚本在单一后端下运行。
    """
    def __init__(self, controller, test_scenarios: list, weights: dict, duration_sec: int = 20,
                 output_folder: str = 'results', gui: bool = False, trajectory: Optional[dict] = None,
                 log_skip: int = 1, in_memory: bool = True, early_stop_rmse: Optional[float] = None,
                 early_min_seconds: float = 4.0, quiet: bool = True):
        self.controller = controller
        self.test_scenarios = test_scenarios
        self.weights = weights
        self.duration_sec = duration_sec
        self.output_folder = output_folder
        self.gui = gui
        self.trajectory = trajectory
        self.CONTROL_FREQ_HZ = 48
        self.CONTROL_TIMESTEP = 1.0 / self.CONTROL_FREQ_HZ
        self.log_skip = max(1, int(log_skip))
        self.in_memory = in_memory
        self.early_stop_rmse = early_stop_rmse
        self.early_min_seconds = max(0.0, early_min_seconds)
        self.quiet = quiet
        self.INITIAL_XYZ = np.array([[0, 0, 1.0]]) if not (trajectory and 'initial_xyz' in trajectory) else np.array([trajectory['initial_xyz']])

    def _generate_trajectory_dataframe(self) -> Optional[pd.DataFrame]:
        if self.trajectory is None:
            return None
        num_steps = int(self.duration_sec * self.CONTROL_FREQ_HZ)
        timestamps = np.linspace(0, self.duration_sec, num=num_steps, endpoint=False)
        traj_type = self.trajectory.get('type')
        params = self.trajectory.get('params', {})
        initial_xyz = self.INITIAL_XYZ[0]
        positions = np.zeros((num_steps, 3))
        if traj_type == 'figure_8':
            A = params.get('A', 0.8); B = params.get('B', 0.5); period = params.get('period', 12.0)
            w = 2*np.pi/period
            for i,t in enumerate(timestamps):
                positions[i,0] = initial_xyz[0] + A*np.sin(w*t)
                positions[i,1] = initial_xyz[1] + B*np.sin(2*w*t)
                positions[i,2] = initial_xyz[2]
        else:
            positions[:] = initial_xyz
        traj_df = pd.DataFrame(positions, columns=['target_x','target_y','target_z'])
        traj_df['timestamp'] = timestamps
        return traj_df.set_index('timestamp')

    def run(self) -> float:
        traj_df = self._generate_trajectory_dataframe()
        effective_dt = self.CONTROL_TIMESTEP * self.log_skip
        reward_calculator = RewardCalculator(weights=self.weights, target_pos=self.INITIAL_XYZ[0], dt=effective_dt, trajectory_df=traj_df)
        # 创建/复用 Isaac Gym 单环境
        global _ENV_SINGLETON
        if _ENV_SINGLETON is None:
            try:
                _ENV_SINGLETON = IsaacGymDroneEnv(num_envs=1, control_freq_hz=self.CONTROL_FREQ_HZ, duration_sec=self.duration_sec, headless=not self.gui)
            except ImportError:
                # Isaac Gym 不可用时，使用简化 CPU 后备环境进行冒烟测试
                _ENV_SINGLETON = SimpleCPUHoverEnv(num_envs=1, control_freq_hz=self.CONTROL_FREQ_HZ, duration_sec=self.duration_sec)
        env = _ENV_SINGLETON
        env.reset()
        # 日志缓冲
        timestamps = []
        xs, ys, zs = [], [], []
        rs, ps, ys_ang = [], [], []
        vxs, vys, vzs = [], [], []
        wxs, wys, wzs = [], [], []
        rpm0, rpm1, rpm2, rpm3 = [], [], [], []

        start = time.time()
        running_sq_error_sum = 0.0; running_count = 0
        disturbance_times: List[float] = [s['time'] for s in self.test_scenarios if s.get('type','PULSE') in ('PULSE','MASS_CHANGE')]

        for i in range(int(self.duration_sec * self.CONTROL_FREQ_HZ)):
            t = i * self.CONTROL_TIMESTEP
            # 状态
            states = env.get_states_batch()
            cur_pos = states['pos'][0].cpu().numpy()
            cur_quat = states['quat'][0].cpu().numpy()
            cur_vel = states['vel'][0].cpu().numpy()
            cur_omega = states['omega'][0].cpu().numpy()

            # 目标
            if traj_df is not None:
                target_pos = traj_df.asof(t)[['target_x','target_y','target_z']].values
            else:
                target_pos = self.INITIAL_XYZ[0]

            rpm, pos_e, rpy_e = self.controller.computeControl(
                control_timestep=self.CONTROL_TIMESTEP,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_omega,
                target_pos=target_pos
            )
            # 施加动作（RPM）
            import torch
            actions = torch.from_numpy(rpm.reshape(1,4)).to(env.device).float()
            _, _, _, _ = env.step(actions)

            if (i % self.log_skip) == 0:
                # 记录日志
                euler = Rotation.from_quat(cur_quat).as_euler('XYZ', degrees=False)
                timestamps.append(t)
                xs.append(cur_pos[0]); ys.append(cur_pos[1]); zs.append(cur_pos[2])
                rs.append(euler[0]); ps.append(euler[1]); ys_ang.append(euler[2])
                vxs.append(cur_vel[0]); vys.append(cur_vel[1]); vzs.append(cur_vel[2])
                wxs.append(cur_omega[0]); wys.append(cur_omega[1]); wzs.append(cur_omega[2])
                rpm0.append(rpm[0]); rpm1.append(rpm[1]); rpm2.append(rpm[2]); rpm3.append(rpm[3])
                # 早停判断
                se = float(np.dot(pos_e, pos_e)); running_sq_error_sum += se; running_count += 1
                if (self.early_stop_rmse is not None) and (t >= self.early_min_seconds) and (running_count>0):
                    rmse = float(np.sqrt(running_sq_error_sum / running_count))
                    if rmse > self.early_stop_rmse:
                        break
            if self.gui:
                time.sleep(max(0.0, t - (time.time() - start)))

        # 组装 DataFrame
        data = np.column_stack([
            np.array(timestamps), np.array(xs), np.array(ys), np.array(zs),
            np.array(rs), np.array(ps), np.array(ys_ang),
            np.array(vxs), np.array(vys), np.array(vzs),
            np.array(wxs), np.array(wys), np.array(wzs),
            np.array(rpm0), np.array(rpm1), np.array(rpm2), np.array(rpm3)
        ])
        cols = ['timestamp','x','y','z','r','p','y_angle','vx','vy','vz','wx','wy','wz','rpm0','rpm1','rpm2','rpm3']
        log_df = pd.DataFrame(data, columns=cols)
        final_reward = reward_calculator.compute_reward(log_df, disturbance_times, verbose=not self.quiet)
        # 单例环境不在此关闭，进程结束由外部回收
        return float(final_reward)
