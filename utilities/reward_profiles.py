"""Reward weight and shaping profiles.

This centralizes different groups of weights so that experiments can switch
between them via a simple CLI flag instead of editing scattered dicts.

Each profile returns two dicts:
- weights: component -> scalar weight (multiplicative factor on shaped term)
- ks: shaping coefficients inside exponential / logistic transforms

Design principles:
- default: balanced, conservative; keeps previous approximate relative scale
- pilight_boost: amplifies components where Pi-Light segmented logic offers
  structural advantages (jerk smoothness, saturation avoidance, gain stability,
  peak error suppression) while slightly de-emphasizing plain position RMSE so
  that early exploration does not overfit a single trajectory.

Adding a new profile: extend PROFILES dict.
"""
from __future__ import annotations
from typing import Dict, Tuple

Weights = Dict[str, float]
Coeffs = Dict[str, float]

# Base (legacy-ish) profile (placeholders; adjust if legacy exact values differ)
_default_weights: Weights = {
    "position_rmse": 1.0,
    "settling_time": 0.6,
    "control_effort": 0.4,
    "smoothness_jerk": 0.6,
    "gain_stability": 0.3,
    "saturation": 0.5,
    "peak_error": 0.7,
    # 新增：高频能量抑制（默认关闭 -> 0.0 保持向后兼容）
    "high_freq": 0.0,
}

_default_ks: Coeffs = {
    # k_??? terms used inside reward shaping (e.g., exp(-k * metric))
    "k_position": 1.2,
    "k_settle": 0.9,
    "k_effort": 0.25,
    "k_jerk": 0.35,
    "k_gain": 0.15,
    "k_sat": 1.0,
    "k_peak": 1.4,
}

# Pi-Light advantage emphasized profile
_pilight_boost_weights: Weights = {
    "position_rmse": 0.85,      # Slightly lower to avoid overfitting early
    "settling_time": 0.65,
    "control_effort": 0.35,     # Encourage using structure even if effort rises mildly
    "smoothness_jerk": 1.10,    # Boost smoothness (segmented gains reduce jerk)
    "gain_stability": 0.80,     # Highlight stability advantages
    "saturation": 0.95,         # Penalize hitting actuator limits
    "peak_error": 1.15,         # Reward suppressing spikes on trajectory transitions
    "high_freq": 0.0,           # 在基础增强里仍默认关闭，可单独选择新 profile
}

_pilight_boost_ks: Coeffs = {
    "k_position": 1.0,   # Slightly softer so other terms matter relatively more
    "k_settle": 1.05,
    "k_effort": 0.20,    # Less punishing, allow transient higher effort
    "k_jerk": 0.55,      # Stronger shaping for jerk reduction
    "k_gain": 0.25,      # Tighter penalty on unstable gain oscillations
    "k_sat": 1.2,        # Faster decay when saturation events occur
    "k_peak": 1.8,       # Stronger penalty for large spikes
    "k_high_freq": 3.0,  # 高频能量 shaping 系数（仅在权重>0时生效）
}

# 新增：强调频域平滑的 profile
_pilight_freq_boost_weights: Weights = {
    # 基于 pilight_boost 做轻微再平衡
    "position_rmse": 0.80,
    "settling_time": 0.60,
    "control_effort": 0.35,
    "smoothness_jerk": 1.00,   # 仍然关注 jerk，但把一部分权重转移给 high_freq
    "gain_stability": 0.75,
    "saturation": 0.90,
    "peak_error": 1.05,
    "high_freq": 1.10,         # 主要新增项：鼓励抑制高频振荡
}

_pilight_freq_boost_ks: Coeffs = {
    "k_position": 1.0,
    "k_settle": 1.05,
    "k_effort": 0.20,
    "k_jerk": 0.55,
    "k_gain": 0.25,
    "k_sat": 1.2,
    "k_peak": 1.8,
    "k_high_freq": 3.0,  # 建议初值，可在未来外显为 CLI 参数
}

# 专门为控制律发现（符号策略综合）设计的 profile
# 核心理念：相比轨迹跟踪DRL，控制律发现更关注鲁棒性和可解释性，
#         而不是过拟合单条轨迹的精确RMSE
_control_law_discovery_weights: Weights = {
    # 降低位置RMSE权重，避免MCTS过度优化单一轨迹而牺牲泛化性
    "position_rmse": 0.60,
    # 强调鲁棒性指标（扰动后恢复速度）
    "settling_time": 1.00,
    # 中等关注控制代价（允许为鲁棒性付出一定代价）
    "control_effort": 0.40,
    # 完全移除平滑性权重（从1.20→0.50→0.20→0.0），避免过度约束大动作探索，让NN自主学习控制策略
    "smoothness_jerk": 0.0,
    # 核心鲁棒性指标：增益稳定性（避免振荡）
    "gain_stability": 1.25,
    # 严格惩罚饱和（饱和意味着控制律在极端情况下失效）
    "saturation": 1.30,
    # 重视峰值误差（体现扰动抑制能力）
    "peak_error": 1.15,
    # 轻度关注高频能量（避免物理不可实现的高频指令）
    "high_freq": 0.80,
}

_control_law_discovery_ks: Coeffs = {
    # 更宽容的位置误差 shaping（允许小误差波动）
    "k_position": 0.8,
    # 强调快速恢复
    "k_settle": 1.3,
    # 中等控制代价敏感度
    "k_effort": 0.18,
    # 进一步放宽平滑性shaping系数（从0.70→0.35→0.20），大幅降低对jerk的敏感度
    "k_jerk": 0.20,
    # 强惩罚增益振荡
    "k_gain": 0.35,
    # 极严格的饱和惩罚
    "k_sat": 1.5,
    # 强惩罚瞬态峰值
    "k_peak": 2.0,
    # 中等高频惩罚
    "k_high_freq": 2.5,
}

# 新增：平滑控制优先 profile（强调 smoothness 和 control effort）
# 适用于需要生成人类可接受、物理可实现、低振动的控制策略场景
_smooth_control_weights: Weights = {
    # 适度降低位置误差权重，平衡跟踪精度与平滑性
    "position_rmse": 0.70,
    # 保持鲁棒性关注
    "settling_time": 0.90,
    # 🔥 显著提升控制代价权重，惩罚过大的控制输出变化
    "control_effort": 0.85,
    # 🔥 重点强调平滑性，抑制加加速度（jerk），生成更平滑的轨迹
    "smoothness_jerk": 1.20,
    # 中等关注增益稳定性
    "gain_stability": 0.80,
    # 严格惩罚饱和
    "saturation": 1.10,
    # 适度关注峰值误差
    "peak_error": 0.95,
    # 强调高频能量抑制，避免高频振荡
    "high_freq": 1.00,
}

_smooth_control_ks: Coeffs = {
    # 稍宽容的位置误差
    "k_position": 0.9,
    # 适中的恢复速度要求
    "k_settle": 1.1,
    # 🔥 强敏感的控制代价 shaping，快速惩罚大幅动作变化
    "k_effort": 0.35,
    # 🔥 强敏感的 jerk shaping，严格抑制加加速度突变
    "k_jerk": 0.65,
    # 适度增益稳定性惩罚
    "k_gain": 0.28,
    # 严格饱和惩罚
    "k_sat": 1.3,
    # 适中峰值惩罚
    "k_peak": 1.6,
    # 强高频惩罚
    "k_high_freq": 3.2,
}

# 平衡型：在不追求PID的前提下，兼顾平滑性、控制代价与跟踪/响应
_balanced_smooth_weights: Weights = {
    "position_rmse": 0.80,     # 维持一定跟踪精度要求
    "settling_time": 1.00,     # 保证有足够的响应速度
    "control_effort": 0.50,    # 中等权重，限制过大控制变化
    "smoothness_jerk": 0.60,   # 中等偏高，鼓励平滑但不过度抑制探索
    "gain_stability": 0.90,    # 稳定性较高权重，减少振荡
    "saturation": 1.20,        # 严格惩罚饱和，保障物理可实现
    "peak_error": 1.00,        # 关注瞬态峰值
    "high_freq": 0.80,         # 抑制高频能量，但不过强
}

_balanced_smooth_ks: Coeffs = {
    "k_position": 1.0,
    "k_settle": 1.1,
    "k_effort": 0.30,    # 略低于 smooth_control，允许必要的响应动作
    "k_jerk": 0.50,      # 略低于 smooth_control，避免过度平滑
    "k_gain": 0.25,
    "k_sat": 1.3,
    "k_peak": 1.8,
    "k_high_freq": 2.5,  # 稍弱于 smooth_control，保留探索弹性
}

PROFILES: Dict[str, Tuple[Weights, Coeffs]] = {
    "default": (_default_weights, _default_ks),
    "pilight_boost": (_pilight_boost_weights, _pilight_boost_ks),
    "pilight_freq_boost": (_pilight_freq_boost_weights, _pilight_freq_boost_ks),
    "control_law_discovery": (_control_law_discovery_weights, _control_law_discovery_ks),
    "smooth_control": (_smooth_control_weights, _smooth_control_ks),
    "balanced_smooth": (_balanced_smooth_weights, _balanced_smooth_ks),
}


def list_profiles() -> Dict[str, Tuple[Weights, Coeffs]]:
    return PROFILES.copy()


def get_reward_profile(name: str) -> Tuple[Weights, Coeffs]:
    if name not in PROFILES:
        raise KeyError(f"Unknown reward profile '{name}'. Available: {list(PROFILES)}")
    weights, ks = PROFILES[name]
    # Return shallow copies to avoid accidental mutation.
    return dict(weights), dict(ks)


def describe_profile(name: str) -> str:
    weights, ks = get_reward_profile(name)
    lines = [f"Reward profile: {name}"]
    lines.append("Weights:")
    for k, v in weights.items():
        lines.append(f"  {k}: {v}")
    lines.append("Coefficients (k_*):")
    for k, v in ks.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

__all__ = [
    "Weights",
    "Coeffs",
    "list_profiles",
    "get_reward_profile",
    "describe_profile",
]