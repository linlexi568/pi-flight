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

PROFILES: Dict[str, Tuple[Weights, Coeffs]] = {
    "default": (_default_weights, _default_ks),
    "pilight_boost": (_pilight_boost_weights, _pilight_boost_ks),
    "pilight_freq_boost": (_pilight_freq_boost_weights, _pilight_freq_boost_ks),
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