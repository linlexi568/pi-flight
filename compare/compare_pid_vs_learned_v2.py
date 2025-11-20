#!/usr/bin/env python3
"""对比 PID 基线 vs 学到的程序（使用 BatchEvaluator，无需 segmented_controller）

配置都写在脚本里，直接运行即可，不传命令行参数。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, '01_pi_flight'))

# ⚠️ 关键：必须先导入 isaacgym，再导入 torch
_GYM_PATH = os.path.join(_ROOT, 'isaacgym', 'python')
if _GYM_PATH not in sys.path:
    sys.path.insert(0, _GYM_PATH)
try:
    from isaacgym import gymapi  # type: ignore
except ImportError:
    print("[WARNING] Isaac Gym 未安装，程序可能无法运行")

import json
from utilities.reward_profiles import get_reward_profile
from utils.batch_evaluation import BatchEvaluator
from core.serialization import load_program_json, deserialize_program

# ==================== 配置区（改这里） ====================

# 学到的程序 JSON 路径
LEARNED_PROGRAM_PATH = os.path.join(_ROOT, "results", "online_best_program.json")

# 评估设置
TRAJECTORY = "figure8"
DURATION = 10  # 秒（与训练保持一致）

# 🔥 奖励 profile - 可选：
#   "safety_first"           - 保守、平滑、节能（强调安全性）
#   "tracking_first"         - 激进跟踪、允许大动作（强调跟踪精度）
#   "balanced"               - 折中方案（综合平衡）
#   "robustness_stability"   - 鲁棒性+稳定性优先（你之前的主实验，强调抗扰动、增益稳定）
#   "control_law_discovery"  - 同 robustness_stability（别名，向后兼容）
#   "smooth_control"         - 平滑控制优先
#   "balanced_smooth"        - 平衡平滑
REWARD_PROFILE = "robustness_stability"  # 🔥 修改这里切换不同奖励策略

ISAAC_NUM_ENVS = 4096
EVAL_REPLICAS = 3  # 每个程序跑几次取平均

# PID 基线程序（手工构造，模拟经典 PD 控制）
# 由于 Isaac Gym 环境中姿态始终保持水平（quat=[0,0,0,1]，无 roll/pitch/yaw 误差），
# 所以基线使用位置误差+速度阻尼来控制推力，使用位置误差的 xy 分量来控制力矩
PID_BASELINE_PROGRAM = [
    {
        "action": [
            {"type": "Binary", "op": "set",
             "left": {"type": "Terminal", "value": "u_fz"},
             "right": {"type": "Binary", "op": "+",
                      "left": {"type": "Terminal", "value": "pos_err_z"},
                      "right": {"type": "Binary", "op": "*",
                               "left": {"type": "Terminal", "value": 0.5},
                               "right": {"type": "Terminal", "value": "vel_z"}}}}
        ]
    },
    {
        "action": [
            {"type": "Binary", "op": "set",
             "left": {"type": "Terminal", "value": "u_tx"},
             "right": {"type": "Binary", "op": "+",
                      "left": {"type": "Binary", "op": "*",
                               "left": {"type": "Terminal", "value": 0.01},
                               "right": {"type": "Terminal", "value": "pos_err_y"}},
                      "right": {"type": "Binary", "op": "*",
                               "left": {"type": "Terminal", "value": 0.005},
                               "right": {"type": "Terminal", "value": "vel_y"}}}}
        ]
    },
    {
        "action": [
            {"type": "Binary", "op": "set",
             "left": {"type": "Terminal", "value": "u_ty"},
             "right": {"type": "Binary", "op": "+",
                      "left": {"type": "Binary", "op": "*",
                               "left": {"type": "Terminal", "value": -0.01},
                               "right": {"type": "Terminal", "value": "pos_err_x"}},
                      "right": {"type": "Binary", "op": "*",
                               "left": {"type": "Terminal", "value": -0.005},
                               "right": {"type": "Terminal", "value": "vel_x"}}}}
        ]
    },
    {
        "action": [
            {"type": "Binary", "op": "set",
             "left": {"type": "Terminal", "value": "u_tz"},
             "right": {"type": "Binary", "op": "*",
                      "left": {"type": "Terminal", "value": -0.001},
                      "right": {"type": "Terminal", "value": "ang_vel_z"}}}
        ]
    }
]

# ==========================================================


def main():
    print("========================================")
    print("对比 PID 基线 vs 学到的程序")
    print("========================================\n")

    print(f"奖励 profile: {REWARD_PROFILE}")
    print(f"轨迹: {TRAJECTORY}, 时长: {DURATION}s, 副本数: {EVAL_REPLICAS}\n")

    # 构造轨迹配置
    if TRAJECTORY == "figure8":
        traj_config = {'type': 'figure_8', 'initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8, 'B': 0.5, 'period': 12}}
    elif TRAJECTORY == "circle":
        traj_config = {'type': 'circle', 'initial_xyz': [0, 0, 0.8], 'params': {'R': 0.9, 'period': 10}}
    elif TRAJECTORY == "helix":
        traj_config = {'type': 'helix', 'initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7, 'period': 10, 'v_z': 0.15}}
    else:
        raise ValueError(f"不支持的轨迹: {TRAJECTORY}")

    # 创建评估器
    evaluator = BatchEvaluator(
        trajectory_config=traj_config,
        duration=DURATION,
        isaac_num_envs=ISAAC_NUM_ENVS,
        device="cuda",
        replicas_per_program=EVAL_REPLICAS,
        reward_reduction="mean",
        reward_profile=REWARD_PROFILE,
        strict_no_prior=True,  # 与训练保持一致
        use_fast_path=False,  # ❗ AST 格式的程序必须禁用 fast_path（ultra_fast_executor 不支持 AST）
    )

    # 1. 评估 PID 基线
    print("[1/2] 评估 PID 基线程序...")
    # 把字典格式转成 AST 对象（deserialize_program 需要 {"rules": [...]} 格式）
    pid_baseline_deserialized = deserialize_program({"rules": PID_BASELINE_PROGRAM})
    print(f"  PID 基线有 {len(pid_baseline_deserialized)} 条规则")
    
    pid_reward = evaluator.evaluate_single(pid_baseline_deserialized)
    print(f"  PID 基线 reward: {pid_reward:.6f}\n")

    # 2. 加载并评估学到的程序
    print("[2/2] 评估学到的程序...")
    if not os.path.isfile(LEARNED_PROGRAM_PATH):
        print(f"  错误：找不到程序文件 {LEARNED_PROGRAM_PATH}")
        sys.exit(1)

    # 用 load_program_json 正确反序列化（把字典转成 AST 对象）
    learned_program = load_program_json(LEARNED_PROGRAM_PATH)
    print(f"  已加载：{LEARNED_PROGRAM_PATH} ({len(learned_program)} 条规则)")
    if learned_program:
        print(f"  第1条规则有 {len(learned_program[0].get('action', []))} 个动作")

    learned_reward = evaluator.evaluate_single(learned_program)
    print(f"  学到的程序 reward: {learned_reward:.6f}\n")

    # 3. 对比
    print("========================================")
    print("对比结果")
    print("========================================")
    print(f"PID 基线:      {pid_reward:.6f}")
    print(f"学到的程序:    {learned_reward:.6f}")
    diff = learned_reward - pid_reward
    print(f"差值 (learned - PID): {diff:+.6f}")
    if diff > 0:
        print("✅ 学到的程序更优！")
    elif diff < 0:
        print("❌ PID 基线更优")
    else:
        print("⚖️ 两者表现相当")
    print("========================================")
    print("\n⚠️  注意：")
    print("学到的程序包含时间算子（ema, rate, delay），这些算子需要跨时间步")
    print("的状态持久化。当前评估框架使用简化的执行路径，时间算子的状态")
    print("可能没有正确维护，导致评估结果可能不准确。")
    print("建议：使用与训练完全一致的评估路径来获得准确的对比结果。")
    print("========================================")


if __name__ == "__main__":
    main()
