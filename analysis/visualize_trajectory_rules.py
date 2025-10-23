"""
可视化测试集轨迹上各段使用的规则分布
在3D轨迹上按规则着色，并标注 stress 干扰事件时间点
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p

# 添加项目根路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.BaseAviary import Physics
from gym_pybullet_drones.utils.enums import DroneModel
from utilities.verify_program import build_trajectory, build_disturbances, build_preset

# 动态加载控制器
import importlib.util
PI_FLIGHT_DIR = os.path.join(ROOT_DIR, '01_pi_flight')
def load_controller():
    init_file = os.path.join(PI_FLIGHT_DIR, '__init__.py')
    spec = importlib.util.spec_from_file_location('piflight_viz', init_file, 
                                                   submodule_search_locations=[PI_FLIGHT_DIR])
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules['piflight_viz'] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return getattr(mod, 'PiLightSegmentedPIDController')
    raise ImportError('Failed to load PiLightSegmentedPIDController')

def load_program_json(path):
    ser_file = os.path.join(PI_FLIGHT_DIR, 'serialization.py')
    spec = importlib.util.spec_from_file_location('piflight_viz.serialization', ser_file)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules['piflight_viz.serialization'] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return mod.load_program_json(path)
    raise ImportError('Failed to load load_program_json')


def run_trajectory_with_rule_tracking(controller, trajectory, disturbances, duration_sec=20):
    """运行仿真并记录每个时间步激活的规则"""
    INITIAL_XYZ = np.array([trajectory['initial_xyz']])
    SIMULATION_FREQ_HZ = 240
    CONTROL_FREQ_HZ = 48
    CONTROL_TIMESTEP = 1.0 / CONTROL_FREQ_HZ
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=INITIAL_XYZ,
        physics=Physics("pyb"),
        pyb_freq=SIMULATION_FREQ_HZ,
        ctrl_freq=CONTROL_FREQ_HZ,
        gui=False,
        record=False
    )
    
    # 预先生成完整轨迹
    from test import SimulationTester
    import tempfile
    temp_dir = tempfile.mkdtemp()
    tester = SimulationTester(controller, [], {}, duration_sec, temp_dir, False, trajectory)
    trajectory_df = tester._generate_trajectory_dataframe()
    
    initial_mass = p.getDynamicsInfo(env.DRONE_IDS[0], -1, physicsClientId=env.CLIENT)[0]
    action = np.zeros((1, 4))
    
    # 记录数据
    positions = []
    target_positions = []
    rules_used = []
    timestamps = []
    
    for i in range(int(duration_sec * CONTROL_FREQ_HZ)):
        obs, _, _, _, _ = env.step(action)
        current_time = i * CONTROL_TIMESTEP
        
        # 应用干扰
        for scenario in disturbances:
            event_type = scenario.get('type', 'PULSE')
            if event_type == 'SUSTAINED_WIND':
                if scenario['start_time'] <= current_time < scenario['end_time']:
                    p.applyExternalForce(
                        objectUniqueId=env.DRONE_IDS[0], 
                        linkIndex=-1, 
                        forceObj=scenario['force'],
                        posObj=[0, 0, 0], 
                        flags=p.WORLD_FRAME, 
                        physicsClientId=env.CLIENT
                    )
            elif event_type == 'GUSTY_WIND':
                if scenario['start_time'] <= current_time < scenario['end_time']:
                    import math
                    base_force = np.array(scenario['base_force'])
                    amp = scenario.get('gust_amplitude', 0.0)
                    freq = scenario.get('gust_frequency', 0.0)
                    gust = amp * math.sin(2 * math.pi * freq * current_time)
                    total_force = base_force + np.array([gust, gust, 0])
                    p.applyExternalForce(
                        objectUniqueId=env.DRONE_IDS[0],
                        linkIndex=-1,
                        forceObj=total_force.tolist(),
                        posObj=[0, 0, 0],
                        flags=p.WORLD_FRAME,
                        physicsClientId=env.CLIENT
                    )
            elif event_type == 'MASS_CHANGE':
                if abs(current_time - scenario['time']) < CONTROL_TIMESTEP / 2:
                    new_mass = initial_mass * scenario['mass_multiplier']
                    p.changeDynamics(
                        bodyUniqueId=env.DRONE_IDS[0],
                        linkIndex=-1,
                        mass=new_mass,
                        physicsClientId=env.CLIENT
                    )
            elif event_type == 'PULSE':
                if abs(current_time - scenario['time']) < CONTROL_TIMESTEP / 2:
                    p.applyExternalForce(
                        objectUniqueId=env.DRONE_IDS[0],
                        linkIndex=-1,
                        forceObj=scenario['force'],
                        posObj=[0, 0, 0],
                        flags=p.WORLD_FRAME,
                        physicsClientId=env.CLIENT
                    )
        
        # 生成目标位置（使用预生成的轨迹数据）
        if trajectory_df is not None:
            target_pos = trajectory_df.asof(current_time)[['target_x', 'target_y', 'target_z']].values
        else:
            target_pos = INITIAL_XYZ[0]
        
        # 计算控制并记录激活的规则
        action[0, :], _, _ = controller.computeControlFromState(
            control_timestep=CONTROL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos
        )
        
        # 记录位置、目标和规则
        positions.append(obs[0][0:3].copy())
        target_positions.append(target_pos.copy())
        rules_used.append(controller.last_rule_name if hasattr(controller, 'last_rule_name') else 'Unknown')
        timestamps.append(current_time)
    
    env.close()
    
    return np.array(positions), np.array(target_positions), rules_used, timestamps


def plot_trajectory_with_rules(positions, target_positions, rules_used, timestamps, 
                                disturbances, trajectory_name, output_path):
    """绘制带规则着色的3D轨迹"""
    # 获取所有唯一规则
    unique_rules = sorted(set(rules_used))
    rule_to_idx = {rule: idx for idx, rule in enumerate(unique_rules)}
    
    # 创建颜色映射
    n_rules = len(unique_rules)
    # 使用推荐 API 获取 colormap
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, max(n_rules, 10)))
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D轨迹图 - 显示实际路径和目标路径
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # 绘制目标轨迹（灰色虚线）
    ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
            'k--', linewidth=1.5, alpha=0.3, label='Target Trajectory')
    
    # 按规则分段绘制实际轨迹
    for i in range(len(positions) - 1):
        rule = rules_used[i]
        color_idx = rule_to_idx[rule]
        ax1.plot(positions[i:i+2, 0], 
                positions[i:i+2, 1], 
                positions[i:i+2, 2],
                color=colors[color_idx % len(colors)],
                linewidth=2,
                alpha=0.8)
    
    # 标记起点和终点
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=100, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='red', s=100, marker='X', label='End', edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.set_title(f'Trajectory: {trajectory_name}\n3D Path Colored by Active Rules', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # XY平面投影
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(len(positions) - 1):
        rule = rules_used[i]
        color_idx = rule_to_idx[rule]
        ax2.plot(positions[i:i+2, 0], 
                positions[i:i+2, 1],
                color=colors[color_idx % len(colors)],
                linewidth=2,
                alpha=0.8)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=80, marker='o', edgecolors='black', linewidths=2)
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=80, marker='X', edgecolors='black', linewidths=2)
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('XY Plane Projection', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 规则使用时间线
    ax3 = fig.add_subplot(2, 2, 3)
    rule_indices = [rule_to_idx[rule] for rule in rules_used]
    ax3.scatter(timestamps, rule_indices, c=rule_indices, cmap=ListedColormap(colors[:n_rules]), 
               s=10, alpha=0.6)
    
    # 标注干扰事件
    for scenario in disturbances:
        event_type = scenario.get('type', '')
        info = scenario.get('info', event_type)
        if event_type in ['SUSTAINED_WIND', 'GUSTY_WIND']:
            ax3.axvspan(scenario['start_time'], scenario['end_time'], 
                       alpha=0.2, color='orange', label=f"{info}")
            ax3.text(scenario['start_time'], n_rules-0.5, info, 
                    rotation=90, fontsize=8, va='bottom')
        elif event_type in ['PULSE', 'MASS_CHANGE']:
            ax3.axvline(scenario['time'], color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax3.text(scenario['time'], n_rules-0.5, info, 
                    rotation=90, fontsize=8, va='bottom')
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Rule Index', fontsize=11)
    ax3.set_title('Rule Activation Timeline\n(with Disturbance Events)', fontsize=12, fontweight='bold')
    ax3.set_yticks(range(n_rules))
    ax3.set_yticklabels(unique_rules, fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 规则使用统计
    ax4 = fig.add_subplot(2, 2, 4)
    rule_counts = {rule: rules_used.count(rule) for rule in unique_rules}
    bars = ax4.barh(unique_rules, [rule_counts[r] for r in unique_rules], 
                    color=colors[:n_rules])
    ax4.set_xlabel('Activation Count', fontsize=11)
    ax4.set_ylabel('Rule', fontsize=11)
    ax4.set_title('Rule Usage Statistics', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 在柱状图上添加百分比
    total = sum(rule_counts.values())
    for i, (bar, rule) in enumerate(zip(bars, unique_rules)):
        width = bar.get_width()
        percentage = 100 * width / total
        ax4.text(width, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[保存] {output_path}")
    plt.close()


def _parse_args():
    ap = argparse.ArgumentParser(description='Visualize rule usage along trajectories with colored segments')
    ap.add_argument('--program', type=str, default=os.path.join(ROOT_DIR, '01_pi_flight', 'results', 'best_program.json'))
    ap.add_argument('--traj-preset', type=str, default='test_challenge',
                    choices=['train_core','test_challenge','full_eval','pi_strong_train','pi_strong_test','test_extreme'])
    ap.add_argument('--disturbance', type=str, default='mild_wind', choices=['none','mild_wind','stress'])
    ap.add_argument('--duration', type=float, default=20.0)
    ap.add_argument('--clip-D', dest='clip_D', type=float, default=None)
    return ap.parse_args()


def main():
    args = _parse_args()
    program_path = args.program
    output_dir = os.path.join(ROOT_DIR, 'results', 'trajectory_rule_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 加载程序
    print(f"[加载程序] {program_path}")
    PiLightSegmentedPIDController = load_controller()
    program = load_program_json(program_path)
    print(f"[程序规则数] {len(program)}")

    # 轨迹集合
    test_trajectories = build_preset(args.traj_preset)

    # 干扰场景
    disturbances = [] if args.disturbance == 'none' else build_disturbances(args.disturbance)

    print(f"\n[干扰场景] {args.disturbance}")
    for d in disturbances:
        print(f"  - {d.get('info', d.get('type'))}")

    # 对每个轨迹运行并可视化
    for traj_name in test_trajectories:
        print(f"\n{'='*60}")
        print(f"[处理轨迹] {traj_name}")
        print('='*60)
        
        trajectory = build_trajectory(traj_name)
        
        # 创建控制器（每个轨迹独立实例，避免状态污染）
        controller = PiLightSegmentedPIDController(
            drone_model=DroneModel.CF2X,
            program=program,
            compose_by_gain=True,
            clip_D=args.clip_D
        )
        
        # 运行仿真
        positions, target_positions, rules_used, timestamps = run_trajectory_with_rule_tracking(
            controller, trajectory, disturbances, duration_sec=int(args.duration)
        )
        
        print(f"[记录步数] {len(positions)}")
        print(f"[激活规则] {set(rules_used)}")
        
        # 可视化
        output_path = os.path.join(output_dir, f'{traj_name}_rule_colored.png')
        plot_trajectory_with_rules(
            positions, target_positions, rules_used, timestamps, disturbances,
            traj_name, output_path
        )
    
    print(f"\n{'='*60}")
    print(f"[完成] 所有轨迹已分析")
    print(f"[输出目录] {output_dir}")
    print('='*60)


if __name__ == '__main__':
    main()
