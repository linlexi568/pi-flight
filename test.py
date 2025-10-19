
import time
import numpy as np
import pybullet as p
import pandas as pd
import os
import math
from typing import Optional

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.BaseAviary import Physics
from gym_pybullet_drones.utils.Logger import Logger
from reward import RewardCalculator

class SimulationTester:
    """
    一个用于测试无人机控制器性能的封装类。
    新版本能够处理多种复杂的、持续性的测试事件，并支持轨迹跟踪。
    """
    
    def __init__(self, controller, test_scenarios: list, weights: dict, duration_sec: int = 25,
                 output_folder: str = 'results', gui: bool = True, trajectory: Optional[dict] = None,
                 log_skip: int = 1, in_memory: bool = False,
                 early_stop_rmse: float | None = None, early_min_seconds: float = 4.0,
                 external_env=None, reuse_reset: bool=False,
                 quiet: bool=False):
        self.controller = controller
        self.test_scenarios = test_scenarios
        self.duration_sec = duration_sec
        self.output_folder = output_folder
        self.gui = gui
        self.weights = weights
        self.trajectory = trajectory
        self.DRONE_MODEL = controller.DRONE_MODEL
        self.SIMULATION_FREQ_HZ = 240
        self.CONTROL_FREQ_HZ = 48
        self.CONTROL_TIMESTEP = 1.0 / self.CONTROL_FREQ_HZ
        self.log_skip = max(1, int(log_skip))
        self.in_memory = in_memory
        self.early_stop_rmse = early_stop_rmse
        self.early_min_seconds = max(0.0, early_min_seconds)
        # 环境复用相关
        self.external_env = external_env
        self.reuse_reset = reuse_reset and (external_env is not None)
        self.quiet = quiet

        # 如果定义了轨迹，则使用轨迹的起始点作为无人机的初始位置
        if self.trajectory and 'initial_xyz' in self.trajectory:
            self.INITIAL_XYZ = np.array([self.trajectory['initial_xyz']])
        else:
            self.INITIAL_XYZ = np.array([[0, 0, 0.5]])

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _generate_trajectory_dataframe(self) -> Optional[pd.DataFrame]:
        """根据 self.trajectory 配置预先生成整个轨迹。"""
        if self.trajectory is None:
            return None

        num_steps = int(self.duration_sec * self.CONTROL_FREQ_HZ)
        timestamps = np.linspace(0, self.duration_sec, num=num_steps, endpoint=False)
        
        traj_type = self.trajectory.get('type')
        params = self.trajectory.get('params', {})
        initial_xyz = self.INITIAL_XYZ[0]
        
        positions = np.zeros((num_steps, 3))

        if traj_type == 'figure_8':
            A = params.get('A', 1.0)
            B = params.get('B', 0.6)
            period = params.get('period', 10)
            w = 2 * np.pi / period
            for i, t in enumerate(timestamps):
                positions[i, 0] = initial_xyz[0] + A * np.sin(w * t)
                positions[i, 1] = initial_xyz[1] + B * np.sin(2 * w * t)
                positions[i, 2] = initial_xyz[2]
        
        elif traj_type == 'helix':
            R = params.get('R', 0.8)
            period = params.get('period', 12)
            v_z = params.get('v_z', 0.1)
            w = 2 * np.pi / period
            for i, t in enumerate(timestamps):
                positions[i, 0] = initial_xyz[0] + R * np.cos(w * t)
                positions[i, 1] = initial_xyz[1] + R * np.sin(w * t)
                positions[i, 2] = initial_xyz[2] + v_z * t

        elif traj_type == 'circle':
            R = params.get('R', 0.9)
            period = params.get('period', 10)
            w = 2 * np.pi / period
            for i, t in enumerate(timestamps):
                positions[i, 0] = initial_xyz[0] + R * np.cos(w * t)
                positions[i, 1] = initial_xyz[1] + R * np.sin(w * t)
                positions[i, 2] = initial_xyz[2]

        elif traj_type == 'square':
            side = params.get('side_len', 1.0)
            period = params.get('period', 12.0)
            corner_hold = params.get('corner_hold', 0.0)
            # 4 边 + 4 次拐角停留（可选）
            pure_edge_time = period - 4 * corner_hold if corner_hold * 4 < period else period * 0.8
            edge_time = pure_edge_time / 4.0
            # 定义四个角点 (绕初始点中心)
            cx, cy, cz = initial_xyz
            half = side / 2.0
            corners = [
                (cx - half, cy - half, cz),
                (cx + half, cy - half, cz),
                (cx + half, cy + half, cz),
                (cx - half, cy + half, cz)
            ]
            def interpolate(p0, p1, alpha):
                return p0[0] + (p1[0]-p0[0])*alpha, p0[1] + (p1[1]-p0[1])*alpha, p0[2] + (p1[2]-p0[2])*alpha
            for i, t in enumerate(timestamps):
                tm = t % period
                acc = 0.0
                pos = None
                for edge_idx in range(4):
                    # corner hold
                    if corner_hold > 0:
                        if acc <= tm < acc + corner_hold:
                            pos = corners[edge_idx]
                            break
                        acc += corner_hold
                    # edge move
                    if acc <= tm < acc + edge_time:
                        alpha = (tm - acc)/edge_time
                        p0 = corners[edge_idx]
                        p1 = corners[(edge_idx+1)%4]
                        pos = interpolate(p0, p1, alpha)
                        break
                    acc += edge_time
                if pos is None:
                    pos = corners[-1]
                positions[i] = pos

        elif traj_type == 'step_hover':
            z2 = params.get('z2', initial_xyz[2] + 0.6)
            switch_time = params.get('switch_time', self.duration_sec/2.0)
            for i, t in enumerate(timestamps):
                positions[i, 0] = initial_xyz[0]
                positions[i, 1] = initial_xyz[1]
                positions[i, 2] = initial_xyz[2] if t < switch_time else z2

        elif traj_type == 'spiral_out':
            R0 = params.get('R0', 0.2)
            k = params.get('k', 0.05)  # 半径增长速率 m/s
            period = params.get('period', 9.0)
            v_z = params.get('v_z', 0.0)
            for i, t in enumerate(timestamps):
                w = 2 * np.pi / period
                r_t = R0 + k * t
                positions[i, 0] = initial_xyz[0] + r_t * np.cos(w * t)
                positions[i, 1] = initial_xyz[1] + r_t * np.sin(w * t)
                positions[i, 2] = initial_xyz[2] + v_z * t
        
        # ===== 新增复杂测试轨迹 =====
        elif traj_type == 'zigzag3d':
            amp = params.get('amplitude', 0.8)
            segments = max(1, int(params.get('segments', 6)))
            z_inc = params.get('z_inc', 0.08)
            period = params.get('period', 14.0)
            seg_time = period / segments
            for i, t in enumerate(timestamps):
                kseg = int((t % period) / seg_time)
                local_t = (t % seg_time) / seg_time  # 0..1
                direction = -1 if (kseg % 2) else 1
                x = direction * amp * (2*local_t - 1)  # 在 [-amp, amp] 来回
                y = (kseg / (segments-1 + 1e-9)) * amp * 0.6 - amp*0.3  # y 逐段推进
                z = initial_xyz[2] + kseg * z_inc
                positions[i] = (initial_xyz[0] + x, initial_xyz[1] + y, z)

        elif traj_type == 'lemniscate3d':
            a = params.get('a', 0.9)
            period = params.get('period', 16.0)
            z_amp = params.get('z_amp', 0.25)
            w = 2 * np.pi / period
            for i, t in enumerate(timestamps):
                theta = w * t
                # Bernoulli lemniscate 近似: x=a*sin(theta), y=a*sin(theta)*cos(theta)
                x = a * math.sin(theta)
                y = a * math.sin(theta) * math.cos(theta)
                z = initial_xyz[2] + z_amp * math.sin(0.5 * theta)
                positions[i] = (initial_xyz[0] + x, initial_xyz[1] + y, z)

        elif traj_type == 'random_waypoints':
            waypoints = params.get('waypoints')
            if not waypoints:
                # 兜底：生成固定集合
                rng = np.random.default_rng(42)
                waypoints = [
                    [rng.uniform(-0.8,0.8), rng.uniform(-0.8,0.8), rng.uniform(0.6,1.2)]
                    for _ in range(6)
                ]
            hold_time = params.get('hold_time', 1.2)
            transition = params.get('transition', 'linear')
            # 预计算每段的时间范围
            schedule = []  # (t_start, t_end, idx, mode)
            t_cursor = 0.0
            for idx in range(len(waypoints)):
                schedule.append((t_cursor, t_cursor+hold_time, idx, 'hold'))
                t_cursor += hold_time
                if idx < len(waypoints)-1:
                    # 过渡保持与 hold_time 同长度，防止总时长过长可以压缩
                    trans_time = hold_time
                    schedule.append((t_cursor, t_cursor+trans_time, idx, 'trans'))
                    t_cursor += trans_time
            total_traj_time = max(t_cursor, 1e-6)
            # 若总时长 < duration，则循环补齐
            def interp(p0, p1, alpha):
                return [p0[i] + (p1[i]-p0[i])*alpha for i in range(3)]
            for i, t in enumerate(timestamps):
                t_mod = t % total_traj_time
                # 找到当前 segment
                seg = None
                for rec in schedule:
                    if rec[0] <= t_mod < rec[1]:
                        seg = rec; break
                if seg is None:
                    seg = schedule[-1]
                t0, t1, idx, mode = seg
                if mode == 'hold':
                    pos = waypoints[idx]
                else:  # trans
                    p0 = waypoints[idx]
                    p1 = waypoints[(idx+1) % len(waypoints)]
                    alpha = (t_mod - t0) / (t1 - t0 + 1e-9)
                    if transition == 'spline':
                        # 简化：使用平滑步进 (3-2alpha)alpha^2 近似 ease-in-out
                        s = alpha*alpha*(3 - 2*alpha)
                        pos = interp(p0, p1, s)
                    else:
                        pos = interp(p0, p1, alpha)
                positions[i] = pos

        elif traj_type == 'spiral_in_out':
            R_in = params.get('R_in', 0.9)
            R_out = params.get('R_out', 0.2)
            period = params.get('period', 14.0)
            z_wave = params.get('z_wave', 0.15)
            half = period / 2.0
            for i, t in enumerate(timestamps):
                tp = t % period
                if tp < half:
                    r = R_in + (R_out - R_in) * (tp / half)
                else:
                    r = R_out + (R_in - R_out) * ((tp - half) / half)
                w = 2 * np.pi / period
                x = r * math.cos(w * t)
                y = r * math.sin(w * t)
                z = initial_xyz[2] + z_wave * math.sin(w * t * 0.5)
                positions[i] = (initial_xyz[0] + x, initial_xyz[1] + y, z)

        elif traj_type == 'stairs':
            levels = params.get('levels', [initial_xyz[2], initial_xyz[2]+0.3, initial_xyz[2]+0.6])
            segment_time = params.get('segment_time', 3.0)
            total_levels = len(levels)
            for i, t in enumerate(timestamps):
                k = int((t / segment_time))
                if k >= total_levels:
                    k = total_levels - 1
                positions[i, 0] = initial_xyz[0]
                positions[i, 1] = initial_xyz[1]
                positions[i, 2] = levels[k]
        
        elif traj_type == 'coupled_surface':
            # x,y 按多频 Lissajous，z 为 x,y 的耦合函数 + 低频起伏
            ax = params.get('ax', 0.9)
            ay = params.get('ay', 0.7)
            f1 = params.get('f1', 1.0)   # 归一化频率 (相对于 duration 内 1 个基频周期)
            f2 = params.get('f2', 2.0)
            phase = params.get('phase', math.pi/3)
            z_base = params.get('z_base', initial_xyz[2])
            z_amp = params.get('z_amp', 0.25)
            surf_amp = params.get('surf_amp', 0.15)
            # 计算基频角速度，使得 f1 对应 duration 内完成 f1 个 2π
            w = 2 * math.pi / self.duration_sec
            for i, t in enumerate(timestamps):
                x = ax * math.sin(w * f1 * t + phase)
                y = ay * math.sin(w * f2 * t)
                # 耦合表面项：sin(x)*cos(y)
                surface = surf_amp * math.sin(x) * math.cos(y)
                z = z_base + z_amp * math.sin(w * 0.5 * t) + surface
                positions[i] = (initial_xyz[0] + x, initial_xyz[1] + y, z)
        
        else:
            print(f"[WARN] 未识别轨迹类型 '{traj_type}'，回退为悬停 (使用 initial 位置)")
            positions[:] = initial_xyz  # 悬停

        traj_df = pd.DataFrame(positions, columns=['target_x', 'target_y', 'target_z'])
        traj_df['timestamp'] = timestamps
        return traj_df.set_index('timestamp')

    def _handle_events(self, env, current_time: float, initial_mass: float):
        """在每个仿真步检查并执行所有当前激活的事件。"""
        for scenario in self.test_scenarios:
            event_type = scenario.get('type', 'PULSE')
            
            if event_type == 'PULSE':
                if abs(current_time - scenario['time']) < self.CONTROL_TIMESTEP / 2:
                    info_txt = scenario.get('info', 'pulse')
                    print(f"[信息] 时间: {current_time:.2f}s, 执行事件: {info_txt}")
                    p.applyExternalForce(objectUniqueId=env.DRONE_IDS[0], linkIndex=-1, forceObj=scenario['force'],
                                         posObj=[0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=env.CLIENT)
            
            elif event_type == 'SUSTAINED_WIND':
                # --- 新增日志 ---
                # 在事件开始时打印一条信息
                if abs(current_time - scenario['start_time']) < self.CONTROL_TIMESTEP / 2:
                    info_txt = scenario.get('info', 'sustained_wind')
                    print(f"[信息] 时间: {current_time:.2f}s, 开始事件: {info_txt}")
                # 持续施加力
                if scenario['start_time'] <= current_time < scenario['end_time']:
                    p.applyExternalForce(objectUniqueId=env.DRONE_IDS[0], linkIndex=-1, forceObj=scenario['force'],
                                         posObj=[0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=env.CLIENT)

            elif event_type == 'GUSTY_WIND':
                # --- 新增日志 ---
                # 在事件开始时打印一条信息
                if abs(current_time - scenario['start_time']) < self.CONTROL_TIMESTEP / 2:
                    info_txt = scenario.get('info', 'gusty_wind')
                    print(f"[信息] 时间: {current_time:.2f}s, 开始事件: {info_txt}")
                # 持续施加力
                if scenario['start_time'] <= current_time < scenario['end_time']:
                    base_force = np.array(scenario['base_force'])
                    amp = scenario.get('gust_amplitude', 0.0) # 使用.get以防不存在
                    freq = scenario.get('gust_frequency', 0.0)
                    gust = amp * math.sin(2 * math.pi * freq * current_time)
                    total_force = base_force + np.array([gust, gust, 0])
                    p.applyExternalForce(objectUniqueId=env.DRONE_IDS[0], linkIndex=-1, forceObj=total_force.tolist(),
                                         posObj=[0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=env.CLIENT)

            elif event_type == 'MASS_CHANGE':
                if abs(current_time - scenario['time']) < self.CONTROL_TIMESTEP / 2:
                    info_txt = scenario.get('info', 'mass_change')
                    print(f"[信息] 时间: {current_time:.2f}s, 执行事件: {info_txt}")
                    new_mass = initial_mass * scenario['mass_multiplier']
                    p.changeDynamics(bodyUniqueId=env.DRONE_IDS[0], linkIndex=-1, mass=new_mass, physicsClientId=env.CLIENT)

    def run(self) -> float:
        """运行完整的仿真和评估流程。"""
        
        trajectory_df = self._generate_trajectory_dataframe()

        # 如果有轨迹，则奖励计算器需要它来计算跟踪误差
        # 根据日志降采样调整 dt（保持 jerk 等导数尺度合理）
        effective_dt = self.CONTROL_TIMESTEP * self.log_skip
        reward_calculator = RewardCalculator(
            weights=self.weights,
            target_pos=self.INITIAL_XYZ[0],
            dt=effective_dt,
            trajectory_df=trajectory_df
        )

        if self.external_env is not None:
            env = self.external_env
            # 复用时手动 reset (软重置: 重置位置/速度/积分项)。BaseAviary 未暴露显式 reset，这里直接 teleport + 清零积分。
            try:
                # 简单 set base pose
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], self.INITIAL_XYZ[0].tolist(), [0,0,0,1], physicsClientId=env.CLIENT)
                p.resetBaseVelocity(env.DRONE_IDS[0], [0,0,0], [0,0,0], physicsClientId=env.CLIENT)
                # 清空控制器内部积分
                if hasattr(self.controller, 'integral_rpy_e'):
                    self.controller.integral_rpy_e[:] = 0.0
                if hasattr(self.controller, 'integral_pos_e'):
                    self.controller.integral_pos_e[:] = 0.0
            except Exception as e:
                print(f"[复用环境重置][WARN] {e}")
        else:
            env = CtrlAviary(drone_model=self.DRONE_MODEL, num_drones=1, initial_xyzs=self.INITIAL_XYZ,
                             physics=Physics("pyb"), pyb_freq=self.SIMULATION_FREQ_HZ, ctrl_freq=self.CONTROL_FREQ_HZ,
                             gui=self.gui, record=False)
        
        initial_mass = p.getDynamicsInfo(env.DRONE_IDS[0], -1, physicsClientId=env.CLIENT)[0]
        logger = Logger(logging_freq_hz=self.CONTROL_FREQ_HZ, num_drones=1, output_folder=self.output_folder, duration_sec=self.duration_sec)
        
        start_time = time.time()
        action = np.zeros((1, 4))
        disturbance_times = [s['time'] for s in self.test_scenarios if s.get('type', 'PULSE') == 'PULSE' or s.get('type') == 'MASS_CHANGE']
        
        running_sq_error_sum = 0.0
        running_count = 0
        for i in range(0, int(self.duration_sec * self.CONTROL_FREQ_HZ)):
            obs, _, _, _, info = env.step(action)
            current_time = i * self.CONTROL_TIMESTEP
            
            self._handle_events(env, current_time, initial_mass)

            # 如果有轨迹，则动态更新目标位置
            if trajectory_df is not None:
                # 使用 asof 来找到最接近当前时间戳的目标点
                target_pos = trajectory_df.asof(current_time)[['target_x', 'target_y', 'target_z']].values
            else:
                # 否则，使用固定的初始位置作为目标
                target_pos = self.INITIAL_XYZ[0]

            action[0, :], _, _ = self.controller.computeControlFromState(
                control_timestep=self.CONTROL_TIMESTEP, state=obs[0], target_pos=target_pos)
            
            if (i % self.log_skip) == 0:
                logger.log(drone=0, timestamp=i/self.CONTROL_FREQ_HZ, state=obs[0])
                # 计算增量 RMSE（对位置或轨迹误差）
                if trajectory_df is not None:
                    err_vec = obs[0][0:3] - target_pos  # x,y,z 在 state 前3位
                else:
                    err_vec = obs[0][0:3] - self.INITIAL_XYZ[0]
                se = float(np.dot(err_vec, err_vec))
                running_sq_error_sum += se
                running_count += 1
                if (self.early_stop_rmse is not None and running_count > 0
                        and current_time >= self.early_min_seconds):
                    running_rmse = math.sqrt(running_sq_error_sum / running_count)
                    if running_rmse > self.early_stop_rmse:
                        # 早停：退出主循环
                        # print 可控：避免噪声（仅必要时输出）
                        # print(f"[EarlyStop] t={current_time:.1f}s rmse={running_rmse:.3f} > {self.early_stop_rmse}")
                        break
            
            if self.gui:
                time.sleep(max(0, (i * self.CONTROL_TIMESTEP) - (time.time() - start_time)))
        
        # --- 日志数据提取 ---
        used = int(logger.counters[0])
        timestamps_arr = logger.timestamps[0, :used]
        states_arr = logger.states[0, :, :used]
        if self.external_env is None:
            env.close()

        if not self.in_memory:
            # 可选：仍然落盘（保持原行为），但为了速度默认 in_memory 使用上面数组即可
            try:
                log_file_path = logger.save()
                # 不再重新读取，直接使用 states_arr/timestamps_arr
            except Exception as e:
                print(f"[WARN] 保存日志失败（继续 in-memory）: {e}")
        try:
            # 组装 DataFrame（匹配原列顺序）
            # states indices: 0:x 1:y 2:z 3:vx 4:vy 5:vz 6:r 7:p 8:yaw 9:wx 10:wy 11:wz 12..15 rpm
            data_matrix = np.column_stack([
                timestamps_arr,
                states_arr[0], states_arr[1], states_arr[2],
                states_arr[6], states_arr[7], states_arr[8],
                states_arr[3], states_arr[4], states_arr[5],
                states_arr[9], states_arr[10], states_arr[11],
                states_arr[12], states_arr[13], states_arr[14], states_arr[15]
            ])
            column_names = ['timestamp','x','y','z','r','p','y_angle','vx','vy','vz','wx','wy','wz','rpm0','rpm1','rpm2','rpm3']
            log_df = pd.DataFrame(data_matrix, columns=column_names)
            final_reward = reward_calculator.compute_reward(log_df, disturbance_times, verbose=not self.quiet)
            return final_reward
        except Exception as e:
            print(f"[错误] 处理内存日志数据失败: {e}")
            return 0.0
