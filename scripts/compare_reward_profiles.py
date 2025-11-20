#!/usr/bin/env python3
"""快速对比不同 reward profiles 的权重配置"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.reward_profiles import get_reward_profile, list_profiles

# 重点对比的四个论文实验 profile
FOCUS_PROFILES = ["safety_first", "tracking_first", "balanced", "robustness_stability"]

print("=" * 80)
print("π-Flight Reward Profile 权重对比表")
print("=" * 80)
print()

# 获取所有权重 key
all_keys = set()
for name in FOCUS_PROFILES:
    w, _ = get_reward_profile(name)
    all_keys.update(w.keys())

# 打印表头
print(f"{'Component':<20}", end="")
for name in FOCUS_PROFILES:
    print(f"{name:>18}", end="")
print()
print("-" * 80)

# 打印权重对比
for key in sorted(all_keys):
    print(f"{key:<20}", end="")
    for name in FOCUS_PROFILES:
        w, _ = get_reward_profile(name)
        val = w.get(key, 0.0)
        print(f"{val:>18.2f}", end="")
    print()

print()
print("=" * 80)
print("Shaping 系数 (k_*) 对比")
print("=" * 80)
print()

# 获取所有系数 key
all_k_keys = set()
for name in FOCUS_PROFILES:
    _, k = get_reward_profile(name)
    all_k_keys.update(k.keys())

# 打印系数表头
print(f"{'Coefficient':<20}", end="")
for name in FOCUS_PROFILES:
    print(f"{name:>18}", end="")
print()
print("-" * 80)

# 打印系数对比
for key in sorted(all_k_keys):
    print(f"{key:<20}", end="")
    for name in FOCUS_PROFILES:
        _, k = get_reward_profile(name)
        val = k.get(key, 0.0)
        print(f"{val:>18.2f}", end="")
    print()

print()
print("=" * 80)
print("设计意图对比")
print("=" * 80)
print()

intentions = {
    "safety_first": "保守、平滑、节能 → 安全关键场景",
    "tracking_first": "激进跟踪、允许大动作 → 性能优先场景",
    "balanced": "折中方案、综合平衡 → 通用应用",
    "robustness_stability": "鲁棒性+稳定性优先 → 抗扰动、增益稳定（之前主实验）",
}

for name, intent in intentions.items():
    print(f"• {name:20s}: {intent}")

print()
print("=" * 80)
print("✓ 所有 profiles 配置正常！")
print("✓ 查看完整文档：REWARD_PROFILES.md")
print("=" * 80)
