import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import sys
import importlib.util
from pathlib import Path

# 确保项目根目录在 sys.path 中 (允许直接运行子目录脚本)
_CUR_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CUR_DIR.parent
_pi_flight = _ROOT_DIR / '01_pi_flight'
_pi_light = _ROOT_DIR / '01_pi_light'
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.BaseAviary import Physics
from gym_pybullet_drones.utils.enums import DroneModel

PiLightSegmentedPIDController = None  # type: ignore
BinaryOpNode = UnaryOpNode = TerminalNode = None  # type: ignore

def _load_pilight_symbols():
    global PiLightSegmentedPIDController, BinaryOpNode, UnaryOpNode, TerminalNode
    # 1) 动态加载 01_pi_light/__init__.py 作为临时模块
    # Prefer new package 01_pi_flight
    init_file = _pi_flight / '__init__.py'
    if init_file.is_file():
        # 标记为包: 提供 submodule_search_locations 以支持 __init__ 中的相对导入 (from .dsl import ...)
        spec = importlib.util.spec_from_file_location(
            'piflight_dynamic', str(init_file), submodule_search_locations=[str(_pi_flight)]
        )
        if spec and spec.loader:  # type: ignore
            module = importlib.util.module_from_spec(spec)
            # 需提前注册到 sys.modules 以便相对导入找到父包
            sys.modules['piflight_dynamic'] = module
            try:
                spec.loader.exec_module(module)  # type: ignore
                PiLightSegmentedPIDController = getattr(module, 'PiLightSegmentedPIDController', None)
                BinaryOpNode = getattr(module, 'BinaryOpNode', None)
                UnaryOpNode = getattr(module, 'UnaryOpNode', None)
                TerminalNode = getattr(module, 'TerminalNode', None)
            except Exception as e:
                print(f"[Warn] 动态加载 01_pi_flight 失败: {e}")
    # Fallback to 01_pi_light
    if PiLightSegmentedPIDController is None:
        init_file = _pi_light / '__init__.py'
        if init_file.is_file():
            spec = importlib.util.spec_from_file_location(
                'pilight_dynamic', str(init_file), submodule_search_locations=[str(_pi_light)]
            )
            if spec and spec.loader:  # type: ignore
                module = importlib.util.module_from_spec(spec)
                sys.modules['pilight_dynamic'] = module
                try:
                    spec.loader.exec_module(module)  # type: ignore
                    PiLightSegmentedPIDController = getattr(module, 'PiLightSegmentedPIDController', None)
                    BinaryOpNode = getattr(module, 'BinaryOpNode', None)
                    UnaryOpNode = getattr(module, 'UnaryOpNode', None)
                    TerminalNode = getattr(module, 'TerminalNode', None)
                except Exception as e:
                    print(f"[Warn] 动态加载 01_pi_light 失败: {e}")
    # 2) 若仍缺失，尝试旧包名 pi_light
    if PiLightSegmentedPIDController is None:
        try:
            import pi_light  # type: ignore
            PiLightSegmentedPIDController = getattr(pi_light, 'PiLightSegmentedPIDController', None)
            BinaryOpNode = getattr(pi_light, 'BinaryOpNode', None)
            UnaryOpNode = getattr(pi_light, 'UnaryOpNode', None)
            TerminalNode = getattr(pi_light, 'TerminalNode', None)
        except Exception:
            pass
    missing = [n for n,v in {
        'PiLightSegmentedPIDController': PiLightSegmentedPIDController,
        'BinaryOpNode': BinaryOpNode,
        'UnaryOpNode': UnaryOpNode,
        'TerminalNode': TerminalNode
    }.items() if v is None]
    if missing:
        raise ImportError(f"无法导入以下符号: {missing}. 请确认 01_pi_light 目录有效或恢复旧 pi_light 包。")

_load_pilight_symbols()

"""dataset_builder.py
离线采集 (state -> 规则触发后实际使用的增益倍率) 数据集，用于训练增益调度网络。
核心输出：
  dataset.npz 包含：
    states: (N, F)
    gains:  (N, 3)  对应 [mP, mI, mD]
    meta:   JSON 字符串（列定义等）
"""

class GainDatasetCollector:
    def __init__(self,
                 program_rules: List[Dict[str, Any]],
                 duration_sec: int = 20,
                 ctrl_freq_hz: int = 48,
                 sim_freq_hz: int = 240,
                 output_path: str = os.path.join('04_nn_baselines','data','gsn_dataset.npz'),
                 drone_model: str = 'cf2x',
                 trajectory: Optional[Dict[str, Any]] = None,
                 gui: bool = False,
                 episodes: int = 1,
                 random_init_xy_range: float = 0.0,
                 random_init_z_range: float = 0.0,
                 augment_factor: int = 0,
                 augment_noise_std: float = 0.0,
                 seed: Optional[int] = None):
        """参数说明:
        episodes: 进行多少独立采集 episode (环境重置)；最终样本数 ~= episodes * duration_sec * ctrl_freq_hz
        random_init_xy_range: 每个 episode 初始 (x,y) 在 [-range, range] 内均匀采样
        random_init_z_range: 初始 z 偏移在 [-range, range]
        augment_factor: 数据增强复制次数；如果 >0，每条原始样本再生成 augment_factor 条带噪版本
        augment_noise_std: 数据增强时对 state 添加的高斯噪声标准差
        seed: 随机种子，保证可复现
        """
        if seed is not None:
            np.random.seed(seed)

        self.program_rules = program_rules
        self.duration_sec = duration_sec
        self.ctrl_freq_hz = ctrl_freq_hz
        self.sim_freq_hz = sim_freq_hz
        self.output_path = output_path
        self.drone_model = DroneModel(drone_model)
        self.trajectory = trajectory
        self.gui = gui
        self.episodes = max(1, episodes)
        self.random_init_xy_range = max(0.0, random_init_xy_range)
        self.random_init_z_range = max(0.0, random_init_z_range)
        self.augment_factor = max(0, augment_factor)
        self.augment_noise_std = max(0.0, augment_noise_std)

        self.state_dim = 0  # 动态推断
        self._records_state: List[np.ndarray] = []
        self._records_gain: List[np.ndarray] = []

    def _build_controller(self):
        # 运行期再使用已动态加载的符号
        return PiLightSegmentedPIDController(drone_model=self.drone_model, program=self.program_rules)  # type: ignore

    def _extract_state(self, env_state: np.ndarray, pos_e: np.ndarray, controller: Any) -> np.ndarray:
        # env_state 顺序参考 logger；这里假设: [x,y,z,r,p,y,..., vx,vy,vz, wx,wy,wz, rpm0..]
        # 安全截取索引
        x, y, z = env_state[0:3]
        r, p_ang, y_ang = env_state[3:6]
        vx, vy, vz = env_state[6:9]
        wx, wy, wz = env_state[9:12]
        # 积分项来自 controller
        int_r = controller.integral_rpy_e[0]
        int_p = controller.integral_rpy_e[1]
        int_yaw = controller.integral_rpy_e[2]
        int_x = controller.integral_pos_e[0]
        int_y = controller.integral_pos_e[1]
        int_z = controller.integral_pos_e[2]
        state_vec = np.array([
            pos_e[0], pos_e[1], pos_e[2],
            vx, vy, vz,
            r, p_ang,
            wx, wy, wz,
            int_r, int_p, int_yaw,
            int_x, int_y, int_z,
            np.mean(controller.P_COEFF_TOR),
            np.mean(controller.I_COEFF_TOR),
            np.mean(controller.D_COEFF_TOR)
        ], dtype=np.float32)
        return state_vec

    def _run_single_episode(self, episode_idx: int):
        steps = int(self.duration_sec * self.ctrl_freq_hz)
        controller = self._build_controller()

        # 随机初始位置
        init_xy = np.zeros(2)
        if self.random_init_xy_range > 0:
            init_xy = np.random.uniform(-self.random_init_xy_range, self.random_init_xy_range, size=2)
        init_z = 0.5
        if self.random_init_z_range > 0:
            init_z += np.random.uniform(-self.random_init_z_range, self.random_init_z_range)
        initial_xyz = np.array([[init_xy[0], init_xy[1], init_z]])

        env = CtrlAviary(drone_model=self.drone_model,
                          num_drones=1,
                          initial_xyzs=initial_xyz,
                          physics=Physics("pyb"),
                          pyb_freq=self.sim_freq_hz,
                          ctrl_freq=self.ctrl_freq_hz,
                          gui=self.gui,
                          record=False)
        action = np.zeros((1,4))
        start_time = time.time()

        for i in range(steps):
            obs, _, _, _, _ = env.step(action)
            target_pos = initial_xyz[0]  # 悬停
            rpm, pos_e, _ = controller.computeControl(
                control_timestep=1.0/self.ctrl_freq_hz,
                cur_pos=obs[0][0:3],
                cur_quat=obs[0][3:7],
                cur_vel=obs[0][10:13],
                cur_ang_vel=obs[0][13:16],
                target_pos=target_pos,
                target_rpy=np.zeros(3),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3)
            )
            action[0, :] = rpm
            state_vec = self._extract_state(obs[0], pos_e, controller)
            self._records_state.append(state_vec)
            last_gain = controller.get_gain_history()[-1]
            self._records_gain.append(np.array([last_gain['P'], last_gain['I'], last_gain['D']], dtype=np.float32))

            if self.gui:
                time.sleep(max(0, (i / self.ctrl_freq_hz) - (time.time() - start_time)))

        env.close()
        print(f"[Episode {episode_idx+1}/{self.episodes}] 样本累计: {len(self._records_state)}")

    def _apply_augmentation(self, states: np.ndarray, gains: np.ndarray):
        if self.augment_factor <= 0 or self.augment_noise_std <= 0:
            return states, gains
        aug_states = [states]
        aug_gains = [gains]
        for k in range(self.augment_factor):
            noise = np.random.normal(0, self.augment_noise_std, size=states.shape).astype(np.float32)
            aug_states.append(states + noise)
            aug_gains.append(gains)  # 不改变标签
        states_cat = np.vstack(aug_states)
        gains_cat = np.vstack(aug_gains)
        return states_cat, gains_cat

    def collect(self):
        for ep in range(self.episodes):
            self._run_single_episode(ep)

        states_arr = np.vstack(self._records_state)
        gains_arr = np.vstack(self._records_gain)
        states_arr, gains_arr = self._apply_augmentation(states_arr, gains_arr)

        self.state_dim = states_arr.shape[1]
        meta = {
            'state_dim': int(self.state_dim),
            'columns': [
                'pos_ex','pos_ey','pos_ez','vx','vy','vz','roll','pitch','wx','wy','wz',
                'int_r','int_p','int_yaw','int_x','int_y','int_z','mean_P','mean_I','mean_D'
            ],
            'episodes': self.episodes,
            'random_init_xy_range': self.random_init_xy_range,
            'random_init_z_range': self.random_init_z_range,
            'augment_factor': self.augment_factor,
            'augment_noise_std': self.augment_noise_std
        }
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        np.savez_compressed(self.output_path, states=states_arr, gains=gains_arr, meta=np.array([str(meta)]))
        print(f"[采集完成] 保存到 {self.output_path} | 原始样本 {len(self._records_state)} | 增强后样本 {len(states_arr)} | state_dim={self.state_dim}")

if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description='离线采集 PI-Light 增益调度数据集')
    parser.add_argument('--duration-sec', type=int, default=5)
    parser.add_argument('--ctrl-freq', type=int, default=48)
    parser.add_argument('--sim-freq', type=int, default=240)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--random-init-xy-range', type=float, default=0.0)
    parser.add_argument('--random-init-z-range', type=float, default=0.0)
    parser.add_argument('--augment-factor', type=int, default=0)
    parser.add_argument('--augment-noise-std', type=float, default=0.0)
    parser.add_argument('--output', type=str, default=os.path.join('04_nn_baselines','data','gsn_dataset.npz'))
    parser.add_argument('--program-json', type=str, default=None, help='包含规则 program 的 JSON 文件路径; 若为空使用内置示例')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()

    # 兼容旧 program 路径: pi_light/results -> 01_pi_flight/results (优先) 或 01_pi_light/results
    if args.program_json and args.program_json.startswith('pi_light'+os.sep):
        legacy_prog = args.program_json
        new_prog_pf = legacy_prog.replace('pi_light'+os.sep, '01_pi_flight'+os.sep, 1)
        new_prog_pl = legacy_prog.replace('pi_light'+os.sep, '01_pi_light'+os.sep, 1)
        if os.path.isfile(new_prog_pf):
            print(f"[LegacyPathWarning] --program-json '{legacy_prog}' -> '{new_prog_pf}'")
            args.program_json = new_prog_pf
        elif os.path.isfile(new_prog_pl):
            print(f"[LegacyPathWarning] --program-json '{legacy_prog}' -> '{new_prog_pl}'")
            args.program_json = new_prog_pl
    if args.program_json and os.path.isfile(args.program_json):
        # 支持两种格式：直接的规则列表 或 包含 rules 字段的序列化 JSON
        with open(args.program_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict) and 'rules' in raw:  # 新序列化格式
            try:
                # 尝试从新编号目录导入反序列化
                try:
                    # Prefer new package path first
                    if str(_pi_flight) not in sys.path:
                        sys.path.insert(0, str(_pi_flight))
                    from serialization import deserialize_program  # type: ignore
                except Exception:
                    try:
                        if str(_pi_light) not in sys.path:
                            sys.path.insert(0, str(_pi_light))
                        from serialization import deserialize_program  # type: ignore
                    except Exception:
                        from pi_light.serialization import deserialize_program  # type: ignore
                program_rules = deserialize_program(raw)
                print(f"[加载] 已从序列化文件读取 {len(program_rules)} 条规则")
            except Exception as e:
                raise RuntimeError(f"无法反序列化程序 {args.program_json}: {e}")
        else:
            program_rules = raw
    else:
        def _default_program():
            # 在调用时检查
            if any(sym is None for sym in (BinaryOpNode, UnaryOpNode, TerminalNode)):
                raise RuntimeError("内置示例规则缺失 DSL 符号，无法构造。")
            cond = BinaryOpNode('>', UnaryOpNode('abs', TerminalNode('pos_err_z')), TerminalNode(0.4))  # type: ignore
            act1 = BinaryOpNode('set', TerminalNode('P'), TerminalNode(1.3))  # type: ignore
            act2 = BinaryOpNode('set', TerminalNode('D'), TerminalNode(0.9))  # type: ignore
            return [{ 'condition': cond, 'action': [act1, act2] }]
        program_rules = _default_program()
    print('[提示] 未提供 --program-json，使用内置示例规则。')

    collector = GainDatasetCollector(
        program_rules=program_rules if isinstance(program_rules, list) else [],
        duration_sec=args.duration_sec,
        ctrl_freq_hz=args.ctrl_freq,
        sim_freq_hz=args.sim_freq,
        output_path=args.output,
        gui=args.gui,
        episodes=args.episodes,
        random_init_xy_range=args.random_init_xy_range,
        random_init_z_range=args.random_init_z_range,
        augment_factor=args.augment_factor,
        augment_noise_std=args.augment_noise_std,
        seed=args.seed
    )
    collector.collect()
