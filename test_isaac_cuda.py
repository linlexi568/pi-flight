#!/usr/bin/env python3
"""测试Isaac Gym环境创建是否有CUDA问题"""

import sys
import os
sys.path.insert(0, '01_pi_flight')

print("Step 1: 导入torch...")
import torch
print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

print("\nStep 2: 创建一些torch tensors...")
x = torch.zeros(256, 6, device='cuda')
print(f"  创建tensor成功: {x.shape}, device={x.device}")

print("\nStep 3: 导入Isaac Gym...")
from envs.isaac_gym_drone_env import IsaacGymDroneEnv

print("\nStep 4: 创建Isaac Gym环境...")
try:
    env = IsaacGymDroneEnv(num_envs=256, control_freq_hz=48, duration_sec=4, headless=True)
    print("  ✅ Isaac Gym环境创建成功!")
    
    print("\nStep 5: Reset环境...")
    obs = env.reset()
    print("  ✅ Reset成功!")
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  position shape: {obs['position'].shape}")
    
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
