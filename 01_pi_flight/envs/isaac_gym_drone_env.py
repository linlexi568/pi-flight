"""
Isaac Gym 无人机环境适配层
提供批量并行的 GPU 加速无人机仿真接口。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os, sys

# Isaac Gym 导入（延迟导入，避免非 GPU 环境报错）
# - 注入 venv/bin 到 PATH（ninja 构建）
# - 自动注入本仓库 isaacgym/python 到 sys.path
# - 自动注入 bindings 到 LD_LIBRARY_PATH（PhysX/rlgpu 等 so）
try:
    _venv_bin = os.path.join(sys.prefix, 'bin')
    if os.path.isdir(_venv_bin):
        os.environ['PATH'] = _venv_bin + os.pathsep + os.environ.get('PATH', '')
    # 计算仓库根路径，并注入 isaacgym/python 与其绑定库路径
    _here = os.path.abspath(os.path.dirname(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, '..', '..'))
    _isaac_py = os.path.join(_repo_root, 'isaacgym', 'python')
    if os.path.isdir(_isaac_py) and _isaac_py not in sys.path:
        sys.path.insert(0, _isaac_py)
    _bindings = os.path.join(_isaac_py, 'isaacgym', '_bindings', 'linux-x86_64')
    if os.path.isdir(_bindings):
        os.environ['LD_LIBRARY_PATH'] = _bindings + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
        # 部分系统需要导出到进程环境后再显式加载
        try:
            import ctypes
            ctypes.CDLL(os.path.join(_bindings, 'libcarb.so'))
        except Exception:
            pass
except Exception:
    pass

try:
    from isaacgym import gymapi, gymtorch, gymutil
    ISAAC_GYM_AVAILABLE = True
except Exception as _ig_e:  # 捕获并打印真实原因
    ISAAC_GYM_AVAILABLE = False
    print("[WARNING] Isaac Gym 导入失败：", repr(_ig_e))
    print("[WARNING] 请确认已安装 isaacgym，并且 LD_LIBRARY_PATH 已包含 _bindings/linux-x86_64 目录")

# 避免在 Isaac Gym 之前导入 torch（官方要求）；到此处再导入 torch
import torch


class IsaacGymDroneEnv:
    """
    批量并行无人机仿真环境（GPU 加速）
    
    特性：
    - 支持 512+ 并行环境实例
    - GPU 加速物理仿真（PhysX）
    - 张量 API（直接输出 PyTorch 张量）
    """
    
    def __init__(
        self,
        num_envs: int = 512,
        device: str = 'cuda:0',
        control_freq_hz: int = 48,
        physics_freq_hz: int = 240,
        duration_sec: float = 20.0,
        initial_height: float = 1.0,
        spacing: float = 3.0,
        headless: bool = True,
        use_gpu: bool = True,
    ):
        """
        初始化 Isaac Gym 批量无人机环境
        
        Args:
            num_envs: 并行环境数量（推荐 256-1024）
            device: PyTorch 设备
            control_freq_hz: 控制频率
            physics_freq_hz: 物理仿真频率
            duration_sec: 每次评估的持续时间
            initial_height: 无人机初始高度
            spacing: 环境间隔距离
            headless: 是否无头模式（无渲染）
            use_gpu: 是否使用 GPU 物理
        """
        if not ISAAC_GYM_AVAILABLE:
            raise ImportError(
                "Isaac Gym 未安装！请从 https://developer.nvidia.com/isaac-gym 下载并安装"
            )
        
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.control_freq = control_freq_hz
        self.physics_freq = physics_freq_hz
        self.dt = 1.0 / physics_freq_hz
        self.control_decimation = physics_freq_hz // control_freq_hz
        self.max_episode_length = int(duration_sec * control_freq_hz)
        self.initial_height = initial_height
        self.spacing = spacing
        
        # 初始化 Gym
        self.gym = gymapi.acquire_gym()
        
        # 配置仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = use_gpu
        sim_params.up_axis = gymapi.UP_AXIS_Z  # Z 轴向上
        
        # PhysX 参数
        sim_params.physx.use_gpu = use_gpu
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1  # TGS solver
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # 重力（恢复真实物理）
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # 创建仿真实例
        compute_device_id = 0 if use_gpu else -1
        graphics_device_id = 0 if not headless else -1
        
        self.sim = self.gym.create_sim(
            compute_device_id,
            graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )
        
        if self.sim is None:
            raise RuntimeError("创建 Isaac Gym 仿真失败！")
        
        # 加载无人机资产
        self._load_drone_asset()
        
        # 创建批量环境
        self._create_envs()
        
        # 准备张量 API
        self.gym.prepare_sim(self.sim)
        self._setup_tensors()
        
        # 状态缓存
        self.reset_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.progress_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_rewards = torch.zeros(num_envs, device=self.device)
        
        print(f"[Isaac Gym] 初始化完成：{num_envs} 个并行环境")
        print(f"  - 设备: {self.device}")
        print(f"  - 控制频率: {control_freq_hz} Hz")
        print(f"  - 物理频率: {physics_freq_hz} Hz")
        print(f"  - GPU 加速: {use_gpu}")
    
    def _load_drone_asset(self):
        """加载无人机模型。
        
        为了完全去除外部仓库依赖，默认使用简化刚体模型。
        如需更精细的无人机 URDF，请将本地 URDF 文件路径接入此处。
        """
        self._create_simplified_drone_asset()
    
    def _create_simplified_drone_asset(self):
        """创建简化的无人机刚体模型（备用方案）"""
        # 创建简单的盒子作为无人机
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.density = 1000.0
        
        # 使用内置形状
        self.drone_asset = self.gym.create_box(
            self.sim, 0.1, 0.1, 0.05, asset_options
        )
    
    def _create_envs(self):
        """创建批量环境实例（优化版：预分配+分块+进度显示）"""
        lower = gymapi.Vec3(-self.spacing, -self.spacing, 0)
        upper = gymapi.Vec3(self.spacing, self.spacing, 2 * self.initial_height)
        
        num_per_row = int(np.sqrt(self.num_envs))
        
        # 预分配列表（避免动态扩容）
        self.envs = [None] * self.num_envs
        self.drone_handles = [None] * self.num_envs
        
        # 预创建pose和刚体属性（避免重复创建对象）
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, self.initial_height)
        pose.r = gymapi.Quat(0, 0, 0, 1)  # 单位四元数
        
        # 分块创建,显示进度（大环境数时重要）
        chunk_size = 2048  # 更大块size减少print开销
        num_chunks = (self.num_envs + chunk_size - 1) // chunk_size
        
        print(f"[Isaac Gym] 创建 {self.num_envs} 个环境 (预计{self.num_envs*0.001:.1f}秒)...")
        import time
        start_time = time.time()
        
        for chunk_idx in range(num_chunks):
            start_i = chunk_idx * chunk_size
            end_i = min((chunk_idx + 1) * chunk_size, self.num_envs)
            chunk_start = time.time()
            
            for i in range(start_i, end_i):
                # 创建环境
                env = self.gym.create_env(self.sim, lower, upper, num_per_row)
                self.envs[i] = env
                
                # 放置无人机
                drone_handle = self.gym.create_actor(
                    env,
                    self.drone_asset,
                    pose,
                    f"drone_{i}",
                    i,  # collision group
                    0   # collision filter
                )
                self.drone_handles[i] = drone_handle
                
                # 设置刚体属性（质量、惯性等）
                props = self.gym.get_actor_rigid_body_properties(env, drone_handle)
                if len(props) > 0:
                    props[0].mass = 0.027  # Crazyflie 质量（kg）
                self.gym.set_actor_rigid_body_properties(env, drone_handle, props)
            
            # 显示详细进度
            chunk_time = time.time() - chunk_start
            total_time = time.time() - start_time
            progress_pct = (end_i / self.num_envs) * 100
            envs_per_sec = end_i / total_time if total_time > 0 else 0
            eta_sec = (self.num_envs - end_i) / envs_per_sec if envs_per_sec > 0 else 0
            
            print(f"  [{end_i:6d}/{self.num_envs}] {progress_pct:5.1f}% | "
                  f"速率: {envs_per_sec:5.0f} envs/s | "
                  f"用时: {total_time:4.1f}s | 预计剩余: {eta_sec:4.1f}s")
    
    def _setup_tensors(self):
        """设置 GPU 张量 API"""
        # 获取刚体状态张量
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, 13)
        
        # 状态分解（便于访问）
        self.pos = self.root_states[:, 0:3]        # 位置 [x, y, z]
        self.quat = self.root_states[:, 3:7]       # 四元数 [qx, qy, qz, qw]
        self.lin_vel = self.root_states[:, 7:10]   # 线速度
        self.ang_vel = self.root_states[:, 10:13]  # 角速度
        
        # 刷新张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 预分配施力/力矩张量（每个环境1个刚体）
        self._num_bodies = 1
        self._rb_forces = torch.zeros((self.num_envs, self._num_bodies, 3), device=self.device, dtype=torch.float32)
        self._rb_torques = torch.zeros((self.num_envs, self._num_bodies, 3), device=self.device, dtype=torch.float32)
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """
        重置指定环境
        
        Args:
            env_ids: 要重置的环境索引（None 表示全部）
        
        Returns:
            obs: 观测字典
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        num_resets = len(env_ids)
        
        # 重置位置
        self.pos[env_ids, 0] = 0.0  # x
        self.pos[env_ids, 1] = 0.0  # y
        self.pos[env_ids, 2] = self.initial_height  # z
        
        # 重置姿态（单位四元数）
        self.quat[env_ids, 0] = 0.0
        self.quat[env_ids, 1] = 0.0
        self.quat[env_ids, 2] = 0.0
        self.quat[env_ids, 3] = 1.0
        
        # 重置速度
        self.lin_vel[env_ids] = 0.0
        self.ang_vel[env_ids] = 0.0
        
        # 应用到仿真
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            num_resets
        )
        
        # 重置缓冲
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.episode_rewards[env_ids] = 0
        
        return self._get_observations()
    
    def step(self, actions: torch.Tensor):
        """
        执行一步控制
        
        Args:
            actions: [num_envs, 4] 张量，表示 [RPM1, RPM2, RPM3, RPM4] 或推力
        
        Returns:
            obs: 观测
            rewards: 奖励
            dones: 终止标志
            info: 额外信息
        """
        # 应用控制力
        # 支持两种输入：
        #  - [N,4] 作为电机 RPM，内部转换为合力/力矩
        #  - [N,6] 直接为 [fx, fy, fz, tx, ty, tz]（机体坐标系）
        if actions.shape[1] == 4:
            forces = self._rpm_to_forces(actions)
        elif actions.shape[1] == 6:
            forces = actions
        else:
            raise ValueError(f"actions 形状非法，期望 [N,4] 或 [N,6]，实际 {tuple(actions.shape)}")
        
        # 执行物理步（内部循环 control_decimation 次）
        for _ in range(self.control_decimation):
            self._apply_forces(forces)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        
        # 刷新状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 更新进度
        self.progress_buf += 1
        
        # 计算奖励和终止
        obs = self._get_observations()
        rewards = self._compute_rewards(obs)
        dones = self._check_termination(obs)
        
        self.episode_rewards += rewards
        
        # 自动重置已终止的环境
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)
        
        return obs, rewards, dones, {}
    
    def _rpm_to_forces(self, actions: torch.Tensor) -> torch.Tensor:
        """
        将 RPM 转换为推力和力矩
        
        Args:
            actions: [num_envs, 4] RPM 命令
        
        Returns:
            forces: [num_envs, 6] 力和力矩 [fx, fy, fz, tx, ty, tz]
        """
        # 将四电机 RPM 转换为机体坐标系的总推力/力矩
        # 模型：
        #   Fz = sum(KF * omega_i^2)
        #   tau_x = L * (T2 - T4)
        #   tau_y = L * (T3 - T1)
        #   tau_z ~ KM * (omega1^2 - omega2^2 + omega3^2 - omega4^2)
        # 调整后的系数，使典型悬停转速在 12k-16k RPM 范围内即可产生足够升力
        KF = 2.8e-08   # N/(rad/s)^2（校准后）
        KM = 1.1e-10   # N*m/(rad/s)^2（校准后，近似）
        L = 0.046      # m，Crazyflie 轴长一半

        omega = actions * (2.0 * np.pi / 60.0)  # RPM -> rad/s
        T = KF * (omega ** 2)  # [N_env, 4]
        Fz = torch.sum(T, dim=1, keepdim=True)
        tau_x = L * (T[:, 1] - T[:, 3]).unsqueeze(1)  # (T2 - T4)
        tau_y = L * (T[:, 2] - T[:, 0]).unsqueeze(1)  # (T3 - T1)
        tau_z = KM * (omega[:, 0] ** 2 - omega[:, 1] ** 2 + omega[:, 2] ** 2 - omega[:, 3] ** 2).unsqueeze(1)

        zeros = torch.zeros((actions.shape[0], 2), device=self.device)
        forces = torch.cat([zeros, Fz, torch.cat([tau_x, tau_y, tau_z], dim=1)], dim=1)
        
        return forces
    
    def _apply_forces(self, forces: torch.Tensor):
        """应用力和力矩到所有无人机（机体坐标系）"""
        # forces: [N,6] -> [fx, fy, fz, tx, ty, tz]
        if forces.shape[0] != self.num_envs or forces.shape[1] != 6:
            raise ValueError(f"forces 形状应为 [num_envs,6]，实际 {tuple(forces.shape)}")
        # 写入张量（每个环境1个刚体）
        self._rb_forces[:, 0, :] = forces[:, 0:3]
        self._rb_torques[:, 0, :] = forces[:, 3:6]
        # 施加（LOCAL_SPACE：机体坐标系）
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._rb_forces),
            gymtorch.unwrap_tensor(self._rb_torques),
            gymapi.LOCAL_SPACE
        )
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        提取观测
        
        Returns:
            obs: 包含位置、速度、姿态等的字典
        """
        # 转换为 NumPy（CPU）以便与现有代码兼容
        obs = {
            'position': self.pos.cpu().numpy(),          # [num_envs, 3]
            'velocity': self.lin_vel.cpu().numpy(),      # [num_envs, 3]
            'orientation': self.quat.cpu().numpy(),      # [num_envs, 4]
            'angular_velocity': self.ang_vel.cpu().numpy()  # [num_envs, 3]
        }
        return obs
    
    def get_obs(self) -> Dict[str, np.ndarray]:
        """
        获取当前观测（不触发reset，用于环境池复用）
        
        Returns:
            obs: 观测字典
        """
        # 刷新状态张量（从GPU同步最新物理状态）
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 返回观测
        return self._get_observations()
    
    def _compute_rewards(self, obs: Dict) -> torch.Tensor:
        """
        计算奖励（批量）
        
        Args:
            obs: 观测字典
        
        Returns:
            rewards: [num_envs] 奖励张量
        """
        # 示例：高度维持任务
        target_height = self.initial_height
        height_error = torch.abs(self.pos[:, 2] - target_height)
        
        # 负误差奖励
        rewards = -height_error
        
        # 额外惩罚：速度过大
        speed_penalty = 0.1 * torch.norm(self.lin_vel, dim=1)
        rewards -= speed_penalty
        
        return rewards
    
    def _check_termination(self, obs: Dict) -> torch.Tensor:
        """
        检查终止条件
        
        Args:
            obs: 观测字典
        
        Returns:
            dones: [num_envs] 布尔张量
        """
        # 坠毁检测
        crashed = (self.pos[:, 2] < 0.1) | (self.pos[:, 2] > 3.0)
        
        # 超时检测
        timeout = self.progress_buf >= self.max_episode_length
        
        dones = crashed | timeout
        return dones
    
    def get_states_batch(self) -> Dict[str, torch.Tensor]:
        """
        获取所有环境的当前状态（张量格式，用于批量控制器）
        
        Returns:
            states: 状态字典（GPU 张量）
        """
        return {
            'pos': self.pos.clone(),
            'vel': self.lin_vel.clone(),
            'quat': self.quat.clone(),
            'omega': self.ang_vel.clone(),
        }
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'gym') and self.gym is not None:
            self.gym.destroy_sim(self.sim)
            print("[Isaac Gym] 环境已关闭")


def test_isaac_gym_env():
    """快速测试脚本"""
    if not ISAAC_GYM_AVAILABLE:
        print("Isaac Gym 未安装，跳过测试")
        return
    
    print("=" * 60)
    print("Isaac Gym 环境测试")
    print("=" * 60)
    
    # 创建环境
    num_envs = 256
    env = IsaacGymDroneEnv(num_envs=num_envs, duration_sec=5.0)
    
    # 重置
    obs = env.reset()
    print(f"初始观测形状: position={obs['position'].shape}")
    
    # 运行短时间仿真
    num_steps = 100
    start_time = time.time()
    
    for step in range(num_steps):
        # 随机动作（RPM）
        actions = torch.rand((num_envs, 4), device=env.device) * 10000 + 5000
        
        obs, rewards, dones, _ = env.step(actions)
        
        if step % 20 == 0:
            print(f"Step {step}: 平均奖励={rewards.mean():.4f}, 终止数={dones.sum()}")
    
    elapsed = time.time() - start_time
    throughput = num_envs * num_steps / elapsed
    
    print("=" * 60)
    print(f"性能统计:")
    print(f"  总步数: {num_steps} steps × {num_envs} envs = {num_steps * num_envs}")
    print(f"  用时: {elapsed:.2f} 秒")
    print(f"  吞吐量: {throughput:.1f} env-steps/秒")
    print(f"  相当于: {throughput / num_envs:.1f} Hz (单环境频率)")
    print("=" * 60)
    
    env.close()


if __name__ == '__main__':
    test_isaac_gym_env()
