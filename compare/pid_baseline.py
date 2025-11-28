#!/usr/bin/env python3
"""
PID 基线评估脚本

使用标准 PID 增益（无程序调参）在一组轨迹上进行评估，
输出格式与 verify_program.py 保持一致，便于对比。
"""
import sys
import os
import argparse
import json
import time
from typing import List, Dict, Any

# 确保能导入 utilities
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utilities.reward_profiles import get_reward_profile


def build_trajectory(name: str) -> Dict[str, Any]:
    """与 verify_program.py 保持一致的轨迹构建"""
    if name == 'figure8':
        return {'type': 'figure_8', 'initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8, 'B': 0.5, 'period': 12}}
    if name == 'helix':
        return {'type': 'helix', 'initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7, 'period': 10, 'v_z': 0.15}}
    if name == 'circle':
        return {'type': 'circle', 'initial_xyz': [0, 0, 0.8], 'params': {'R': 0.9, 'period': 10}}
    if name == 'hover':
        return {'type': 'hover', 'initial_xyz': [0, 0, 1.0], 'params': {}}
    # 可继续添加更多轨迹
    raise ValueError(f"Unknown trajectory: {name}")


def build_preset(preset: str) -> List[str]:
    """预设轨迹集合"""
    if preset == 'basic':
        return ['hover', 'circle', 'figure8']
    if preset == 'test_challenge':
        return ['circle', 'figure8', 'helix']
    if preset == 'full':
        return ['hover', 'circle', 'figure8', 'helix']
    raise ValueError(f"Unknown preset: {preset}")


def harmonic_mean(values: List[float]) -> float:
    """调和平均"""
    if not values:
        return 0.0
    return len(values) / sum(1.0 / (v + 1e-9) for v in values)


def main():
    parser = argparse.ArgumentParser(description='PID Baseline Evaluation')
    parser.add_argument('--traj_preset', type=str, default='basic',
                        choices=['basic', 'test_challenge', 'full'])
    parser.add_argument('--traj_list', type=str, nargs='*', default=None)
    parser.add_argument('--aggregate', type=str, default='harmonic',
                        choices=['mean', 'min', 'harmonic'])
    parser.add_argument('--duration', type=int, default=20)
    parser.add_argument('--reward_profile', type=str, default='safe_control_tracking', choices=['safe_control_tracking'])
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 文件路径（可选）')
    args = parser.parse_args()

    # 获取奖励权重
    weights, ks = get_reward_profile(args.reward_profile)
    
    # 构建轨迹列表
    if args.traj_list:
        traj_names = args.traj_list
    else:
        traj_names = build_preset(args.traj_preset)
    
    trajectories = [build_trajectory(name) for name in traj_names]
    
    print(f"[PID Baseline] 评估配置:")
    print(f"  - 轨迹: {traj_names}")
    print(f"  - 聚合方式: {args.aggregate}")
    print(f"  - 奖励配置: {args.reward_profile}")
    print(f"  - 持续时间: {args.duration}s")
    
    # 导入仿真测试器
    try:
        from utilities.isaac_tester import SimulationTester
    except ImportError:
        print("[ERROR] 无法导入 SimulationTester，请确保 Isaac Gym 已安装")
        sys.exit(1)
    
    # 导入标准 PID 控制器（无程序增强）
    try:
        # 尝试从 01_pi_flight 导入基础 PID
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "local_pid",
            os.path.join(_ROOT, "01_pi_flight", "utils", "segmented_controller.py")
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # 使用父类 SimplePIDControl 作为基线（不加载程序）
            # 但 segmented_controller 继承自 local_pid，需要找到基类
            # 简化：直接用空程序的 PiLightSegmentedPIDController
            PIDController = mod.PiLightSegmentedPIDController
    except Exception as e:
        print(f"[ERROR] 无法加载 PID 控制器: {e}")
        sys.exit(1)
    
    scores: List[float] = []
    per_traj: List[Dict[str, Any]] = []
    
    start_time = time.time()
    
    for traj in trajectories:
        traj_name = traj['type']
        print(f"\n[PID Baseline] 评估轨迹: {traj_name}")
        
        # 创建标准 PID 控制器（无程序 = 空 program）
        ctrl = PIDController(
            drone_model="cf2x",
            program=[],  # 空程序 = 使用默认 PID 增益
            suppress_init_print=True
        )
        
        # 运行仿真
        tester = SimulationTester(
            ctrl,
            disturbances=[],  # PID 基线无扰动
            weights=weights,
            duration_sec=args.duration,
            output_folder=os.path.join(_ROOT, 'compare', 'logs', 'pid_runs'),
            gui=False,
            trajectory=traj,
            log_skip=2,
            in_memory=True,
            quiet=True
        )
        
        reward = tester.run()
        scores.append(float(reward))
        per_traj.append({'traj': traj_name, 'reward': float(reward)})
        print(f"  -> 奖励: {reward:.6f}")
    
    elapsed = time.time() - start_time
    
    # 聚合
    if args.aggregate == 'mean':
        agg = sum(scores) / len(scores)
    elif args.aggregate == 'min':
        agg = min(scores)
    else:
        agg = harmonic_mean(scores)
    
    print(f"\n{'='*60}")
    print(f"[PID Baseline] 聚合得分 ({args.aggregate}): {agg:.6f}")
    print(f"[PID Baseline] 总耗时: {elapsed:.1f}s")
    print(f"{'='*60}")
    print("\n逐轨迹结果:")
    for item in per_traj:
        print(f"  {item['traj']:20s}: {item['reward']:.6f}")
    
    # 可选：保存到 JSON
    if args.output:
        result = {
            'method': 'PID_baseline',
            'aggregate_score': float(agg),
            'aggregate_method': args.aggregate,
            'total_time_sec': float(elapsed),
            'trajectories': traj_names,
            'per_traj': per_traj,
            'config': {
                'duration': args.duration,
                'reward_profile': args.reward_profile,
                'traj_preset': args.traj_preset
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[PID Baseline] 结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
