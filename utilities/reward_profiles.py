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

# 🔥 专门为控制律发现（符号策略综合）设计的 profile
# 核心理念：鲁棒性 + 稳定性优先
#   - 相比轨迹跟踪DRL，控制律发现更关注鲁棒性和可解释性
#   - 不过拟合单条轨迹的精确RMSE，而是追求泛化能力
#   - 强调增益稳定性、扰动恢复、饱和避免（核心鲁棒性指标）
#   - 适度牺牲跟踪精度和平滑性，换取更强的抗扰动能力
_robustness_stability_weights: Weights = {
    # 降低位置RMSE权重，避免MCTS过度优化单一轨迹而牺牲泛化性
    "position_rmse": 0.60,
    # 🔥 强调鲁棒性指标：扰动后恢复速度（settling time）
    "settling_time": 1.00,
    # 中等关注控制代价（允许为鲁棒性付出一定代价）
    "control_effort": 0.40,
    # 完全移除平滑性权重，避免过度约束大动作探索，让NN自主学习控制策略
    "smoothness_jerk": 0.0,
    # 🔥 核心鲁棒性指标：增益稳定性（避免振荡、参数敏感性低）
    "gain_stability": 1.25,
    # 🔥 严格惩罚饱和（饱和意味着控制律在极端情况下失效）
    "saturation": 1.30,
    # 🔥 重视峰值误差（体现扰动抑制能力）
    "peak_error": 1.15,
    # 轻度关注高频能量（避免物理不可实现的高频指令）
    "high_freq": 0.80,
}

_robustness_stability_ks: Coeffs = {
    # 更宽容的位置误差 shaping（允许小误差波动）
    "k_position": 0.8,
    # 🔥 强调快速恢复
    "k_settle": 1.3,
    # 中等控制代价敏感度
    "k_effort": 0.18,
    # 进一步放宽平滑性shaping系数，大幅降低对jerk的敏感度
    "k_jerk": 0.20,
    # 🔥 强惩罚增益振荡
    "k_gain": 0.35,
    # 🔥 极严格的饱和惩罚
    "k_sat": 1.5,
    # 🔥 强惩罚瞬态峰值
    "k_peak": 2.0,
    # 中等高频惩罚
    "k_high_freq": 2.5,
}

# 向后兼容别名（保留原名）
_control_law_discovery_weights = _robustness_stability_weights
_control_law_discovery_ks = _robustness_stability_ks

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

# =============================================================================
# 🔥 新增：专为论文实验设计的三版奖励 profile
# =============================================================================

# 1️⃣ Safety-First：保守、平滑、节能
# 设计意图：
#   - 高度重视安全性（不炸机、不饱和、不振荡）
#   - 强调控制平滑性（低 jerk、低高频能量）
#   - 允许适度的位置误差，换取更稳定的控制行为
#   - 适用于安全关键应用、演示、以及作为 baseline 对比
_safety_first_weights: Weights = {
    "position_rmse": 0.70,        # 中等偏低：不过分追求误差，避免激进控制
    "settling_time": 0.80,        # 中等：保证一定响应速度
    "control_effort": 0.85,       # 🔥 高权重：严格限制控制幅度
    "smoothness_jerk": 1.30,      # 🔥 极高权重：强调平滑、抑制抖动
    "gain_stability": 1.00,       # 高权重：避免增益振荡
    "saturation": 1.50,           # 🔥 极高权重：几乎不允许饱和
    "peak_error": 0.90,           # 中等：关注但不强迫
    "high_freq": 1.20,            # 🔥 高权重：强抑制高频振荡
}

_safety_first_ks: Coeffs = {
    "k_position": 0.85,           # 较宽容的位置误差 shaping
    "k_settle": 1.0,
    "k_effort": 0.45,             # 🔥 强敏感：快速惩罚大动作
    "k_jerk": 0.75,               # 🔥 强敏感：严格抑制加加速度
    "k_gain": 0.30,
    "k_sat": 1.6,                 # 🔥 极严格：饱和立即重罚
    "k_peak": 1.5,
    "k_high_freq": 3.5,           # 🔥 强惩罚高频
}

# 2️⃣ Tracking-First：激进跟踪、允许大动作
# 设计意图：
#   - 极度重视轨迹跟踪精度（低 RMSE、低峰值误差、快速 settling）
#   - 大幅降低对控制代价和平滑性的惩罚
#   - 允许频繁打满、高频动作，只要能跟上轨迹
#   - 适用于性能优先场景、与 PID/PPO 对比时的"上限"展示
_tracking_first_weights: Weights = {
    "position_rmse": 1.50,        # 🔥 极高权重：核心目标
    "settling_time": 1.20,        # 🔥 高权重：快速响应
    "control_effort": 0.20,       # 🔥 极低：允许大动作
    "smoothness_jerk": 0.15,      # 🔥 极低：允许抖动
    "gain_stability": 0.40,       # 低：允许一定振荡
    "saturation": 0.30,           # 🔥 极低：可以频繁饱和
    "peak_error": 1.40,           # 🔥 高权重：严格压制瞬态误差
    "high_freq": 0.25,            # 🔥 极低：允许高频指令
}

_tracking_first_ks: Coeffs = {
    "k_position": 1.5,            # 🔥 强敏感：位置误差快速放大
    "k_settle": 1.4,              # 🔥 强敏感：快速收敛要求
    "k_effort": 0.12,             # 🔥 极宽容：大动作几乎不惩罚
    "k_jerk": 0.18,               # 🔥 极宽容：jerk 几乎不管
    "k_gain": 0.15,               # 宽容振荡
    "k_sat": 0.5,                 # 🔥 极宽容：饱和惩罚很轻
    "k_peak": 2.2,                # 🔥 强敏感：峰值误差严厉打击
    "k_high_freq": 1.5,           # 宽容高频
}

# 3️⃣ Balanced：折中方案
# 设计意图：
#   - 在跟踪精度和控制平滑之间取平衡
#   - 各项权重居中，适合作为"主实验结果"展示
#   - 体现 π-Flight 在多目标优化下的综合优势
#   - 与 PID 和 PPO 的对比中，展示"既不过分保守也不过分激进"的中庸之道
_balanced_weights: Weights = {
    "position_rmse": 1.00,        # 标准权重
    "settling_time": 0.90,        # 标准权重
    "control_effort": 0.50,       # 中等：限制但不过分
    "smoothness_jerk": 0.70,      # 中等偏高：鼓励平滑
    "gain_stability": 0.80,       # 中等偏高：避免振荡
    "saturation": 1.00,           # 标准：不鼓励饱和
    "peak_error": 1.00,           # 标准：关注峰值
    "high_freq": 0.70,            # 中等：抑制但不过强
}

_balanced_ks: Coeffs = {
    "k_position": 1.1,
    "k_settle": 1.15,
    "k_effort": 0.28,
    "k_jerk": 0.48,
    "k_gain": 0.26,
    "k_sat": 1.1,
    "k_peak": 1.7,
    "k_high_freq": 2.8,
}

# =============================================================================

PROFILES: Dict[str, Tuple[Weights, Coeffs]] = {
    "default": (_default_weights, _default_ks),
    "pilight_boost": (_pilight_boost_weights, _pilight_boost_ks),
    "pilight_freq_boost": (_pilight_freq_boost_weights, _pilight_freq_boost_ks),
    # 🔥 鲁棒性+稳定性优先（原 control_law_discovery，保留两个名字）
    "robustness_stability": (_robustness_stability_weights, _robustness_stability_ks),
    "control_law_discovery": (_control_law_discovery_weights, _control_law_discovery_ks),  # 别名，向后兼容
    "smooth_control": (_smooth_control_weights, _smooth_control_ks),
    "balanced_smooth": (_balanced_smooth_weights, _balanced_smooth_ks),
    # 🔥 论文实验专用三大 profile
    "safety_first": (_safety_first_weights, _safety_first_ks),
    "tracking_first": (_tracking_first_weights, _tracking_first_ks),
    "balanced": (_balanced_weights, _balanced_ks),
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