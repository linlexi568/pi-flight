"""
GPU端程序执行器 - 消除CPU-GPU传输瓶颈

关键优化:
1. 所有计算在GPU上完成（PyTorch）
2. 避免.cpu().numpy()转换
3. 预编译常量程序为GPU tensor
4. 批量向量化操作
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional


class GPUProgramExecutor:
    """GPU端批量程序执行器"""
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device)
        self.program_cache = {}  # 缓存预编译程序
        
    def compile_constant_programs(self, programs: List[List[Dict[str, Any]]]) -> Optional[torch.Tensor]:
        """
        预编译常量程序为GPU tensor
        
        Args:
            programs: 程序列表，每个程序是规则列表
            
        Returns:
            forces_tensor: [n_programs, 4] 的GPU tensor (fz, tx, ty, tz)
                          如果有非常量程序则返回None
        """
        forces_list = []
        
        for prog in programs:
            # 检查是否为常量程序 (所有规则都是 set u_* = const)
            forces = self._extract_constant_forces(prog)
            if forces is None:
                return None  # 包含非常量程序，无法批量编译
            forces_list.append(forces)
        
        # 转为GPU tensor
        forces_tensor = torch.tensor(forces_list, device=self.device, dtype=torch.float32)
        return forces_tensor
    
    def _extract_constant_forces(self, program: List[Dict[str, Any]]) -> Optional[List[float]]:
        """
        从程序中提取常量力 (fz, tx, ty, tz)
        
        Returns:
            [fz, tx, ty, tz] 或 None (如果不是常量程序)
        """
        forces = [0.0, 0.0, 0.0, 0.0]  # fz, tx, ty, tz
        var_map = {'u_fz': 0, 'u_tx': 1, 'u_ty': 2, 'u_tz': 3}
        
        for rule in program:
            if rule.get('op') != 'set':
                return None  # 非set操作
                
            var = rule.get('var')
            if var not in var_map:
                return None  # 未知变量
                
            expr = rule.get('expr', {})
            if expr.get('type') != 'const':
                return None  # 非常量表达式
                
            value = expr.get('value', 0.0)
            forces[var_map[var]] = float(value)
        
        return forces
    
    def apply_constant_forces_gpu(
        self, 
        forces_tensor: torch.Tensor,
        batch_size: int,
        num_envs: int
    ) -> torch.Tensor:
        """
        在GPU上批量应用常量力
        
        Args:
            forces_tensor: [batch_size, 4] 预编译的力
            batch_size: 程序数量
            num_envs: 总环境数
            
        Returns:
            actions: [num_envs, 6] GPU tensor (fx=0, fy=0, fz, tx, ty, tz)
        """
        actions = torch.zeros((num_envs, 6), device=self.device, dtype=torch.float32)
        
        # 将力复制到actions的对应位置 (fz在index 2, tx/ty/tz在index 3/4/5)
        actions[:batch_size, 2:6] = forces_tensor
        
        return actions
    
    def quat_to_rpy_gpu(self, quat: torch.Tensor) -> torch.Tensor:
        """
        在GPU上将四元数转换为欧拉角 (Roll-Pitch-Yaw)
        
        Args:
            quat: [batch_size, 4] GPU tensor (x, y, z, w)
            
        Returns:
            rpy: [batch_size, 3] GPU tensor (roll, pitch, yaw)
        """
        # 四元数分量
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # 裁剪避免asin的域错误
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=1)


def test_gpu_executor():
    """测试GPU执行器性能"""
    import time
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    executor = GPUProgramExecutor(device)
    
    # 创建测试程序 (常量程序)
    test_program = [
        {'op': 'set', 'var': 'u_fz', 'expr': {'type': 'const', 'value': 0.5}},
        {'op': 'set', 'var': 'u_tx', 'expr': {'type': 'const', 'value': 0.0}},
        {'op': 'set', 'var': 'u_ty', 'expr': {'type': 'const', 'value': 0.0}},
        {'op': 'set', 'var': 'u_tz', 'expr': {'type': 'const', 'value': 0.0}},
    ]
    
    batch_size = 2000
    programs = [test_program] * batch_size
    
    # 编译
    print("编译常量程序...")
    forces = executor.compile_constant_programs(programs)
    print(f"✅ 编译完成: {forces.shape}")
    
    # 测试应用力
    num_envs = 16384
    print(f"\n测试批量应用力 ({batch_size}程序 → {num_envs}环境)...")
    
    times = []
    for _ in range(100):
        t0 = time.time()
        actions = executor.apply_constant_forces_gpu(forces, batch_size, num_envs)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1-t0)*1000)
    
    avg_time = np.mean(times)
    print(f"平均耗时: {avg_time:.3f}ms")
    
    # 测试四元数转换
    print(f"\n测试四元数→RPY转换 ({batch_size}个四元数)...")
    quat = torch.randn(batch_size, 4, device=device)
    quat = quat / torch.norm(quat, dim=1, keepdim=True)
    
    times = []
    for _ in range(100):
        t0 = time.time()
        rpy = executor.quat_to_rpy_gpu(quat)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1-t0)*1000)
    
    avg_time = np.mean(times)
    print(f"平均耗时: {avg_time:.3f}ms")
    
    print(f"\n✅ GPU执行器测试完成!")


if __name__ == '__main__':
    test_gpu_executor()
