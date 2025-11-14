"""Stepwise reward calculator for online AlphaZero-style program synthesis.

近似 reward.py 的多维度成分,支持在线逐步计算,避免缓存整段 episode。

包含组件 (与 control_law_discovery profile 对齐):
- position_rmse (逐步近似 -> 当前位置误差)
- settling_time (代理: 误差从高到低下降的速度趋势奖励)
- control_effort (动作差分的二范数)
- smoothness_jerk (速度二阶差分近似 jerk)
- gain_stability (最近若干步动作变化的方差)
- saturation (单步是否触达电机上限/推力上限)
- peak_error (记录全局最大误差,最终归一惩罚)
- high_freq (角速度差分能量)

返回: 每步 composite reward (加权和) + 可选分量字典；最终调用 finalize() 增加一次 peak / settling bonus/penalty。

设计原则:
1. 所有度量映射到 [0,1] 或小负值稳定区间,避免梯度爆炸。
2. 采用 exp(-k * metric) 形状函数保持与 reward.py 一致的单调性。
3. 计算量 O(num_envs)；历史缓存窗口长度 <=3,常数内存。

引用理论 (详见 docs/reward_theory.md):
- 位置误差与 RMSE: 经典轨迹跟踪控制 (Åström & Murray, 2021)
- Settling time: 二阶系统瞬态指标 (Ogata, 2010)
- 控制能量: L2 输入代价 (Optimal Control, Lewis 2012)
- Jerk 抑制: 人类舒适/结构疲劳 (Flash & Hogan 1985)
- 增益稳定性: 动力学一致性/抗振 (Skogestad 2005)
- 执行器饱和: Anti-windup 设计 (Åström & Hägglund 2006)
- 峰值误差: 最坏情况下鲁棒性 (Zhou & Doyle 1998)
- 高频能量: 振动与结构耦合 (Scherer 2019)
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import math

class StepwiseRewardCalculator:
    def __init__(self, weights: Dict[str, float], ks: Dict[str, float], dt: float, num_envs: int, device: str = 'cpu',
                 rpm_max: float = 21702.0, saturation_ratio: float = 0.95):
        self.w = weights
        self.k = ks
        self.dt = dt
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.rpm_max = rpm_max
        self.sat_thr = saturation_ratio * rpm_max
        # 历史缓存
        self.vel_hist = []              # list[Tensor [N,3]] 最近 3 个速度
        self.last_actions: Optional[torch.Tensor] = None  # [N, A]
        self.action_var_accum = torch.zeros(num_envs, device=self.device)  # 累积动作变化方差估计
        self.action_var_count = torch.zeros(num_envs, device=self.device)
        self.max_pos_err = torch.zeros(num_envs, device=self.device)
        self.prev_pos_err = torch.zeros(num_envs, device=self.device)
        self.settle_progress = torch.zeros(num_envs, device=self.device)  # 误差下降趋势积分
        self.steps = 0
        # 组件累积（记录加权后的分量总和，便于批次结束时导出指标）
        self._comp_accum = {
            'position_rmse': torch.zeros(num_envs, device=self.device),
            'settling_time': torch.zeros(num_envs, device=self.device),
            'control_effort': torch.zeros(num_envs, device=self.device),
            'smoothness_jerk': torch.zeros(num_envs, device=self.device),
            'gain_stability': torch.zeros(num_envs, device=self.device),
            'saturation': torch.zeros(num_envs, device=self.device),
            'peak_error': torch.zeros(num_envs, device=self.device),
            'high_freq': torch.zeros(num_envs, device=self.device),
        }

    @staticmethod
    def _exp_shape(metric: torch.Tensor, k: float) -> torch.Tensor:
        # 防止数值下溢: 限制 metric 范围
        # 返回 -1 * (1 - exp(-k*metric)) 使其变为负惩罚
        # metric=0 → reward=0 (理想)
        # metric→∞ → reward→-1 (最差单分量)
        return -1.0 * (1.0 - torch.exp(-k * torch.clamp(metric, min=0.0, max=1e6)))

    def compute_step(self,
                     pos: torch.Tensor,           # [N,3]
                     target: torch.Tensor,        # [3]
                     vel: torch.Tensor,           # [N,3]
                     omega: torch.Tensor,         # [N,3]
                     actions: torch.Tensor,       # [N,4 or 6]
                     done_mask: torch.Tensor      # [N] bool (已结束环境不再计入)
                     ) -> torch.Tensor:
        """计算单步奖励(对仍活跃环境)。"""
        self.steps += 1
        active = ~done_mask  # bool
        active_f = active.float()
        # 位置误差
        pos_err = torch.norm(pos - target.view(1,3), dim=1)  # [N]
        self.max_pos_err = torch.maximum(self.max_pos_err, pos_err)
        # 误差趋势 (settling proxy): 如果当前误差 < 上一步误差则奖励 +delta
        delta_err = self.prev_pos_err - pos_err
        self.settle_progress += torch.clamp(delta_err, min=0.0)  # 只积正向下降
        self.prev_pos_err = pos_err.detach()

        # Position component
        r_pos = self._exp_shape(pos_err, self.k.get('k_position', 1.0)) * active_f

        # Settling proxy component (归一化: 累积下降 / (1+步数))
        settle_norm = self.settle_progress / (1.0 + self.steps)
        r_settle = self._exp_shape(1.0 - settle_norm, self.k.get('k_settle', 1.0)) * active_f

        # Control effort: action diff L2
        if self.last_actions is None:
            action_diff = torch.zeros_like(actions)
        else:
            action_diff = actions - self.last_actions
        self.last_actions = actions.detach()
        effort = torch.sqrt(torch.clamp(torch.sum(action_diff**2, dim=1), min=0.0))
        r_effort = self._exp_shape(effort, self.k.get('k_effort', 0.2)) * active_f

        # Jerk: 根据速度历史二阶差分
        self.vel_hist.append(vel.clone())
        if len(self.vel_hist) > 3:
            self.vel_hist.pop(0)
        if len(self.vel_hist) >= 3:
            acc1 = (self.vel_hist[1] - self.vel_hist[0]) / self.dt
            acc2 = (self.vel_hist[2] - self.vel_hist[1]) / self.dt
            jerk = torch.sqrt(torch.sum(((acc2 - acc1) / self.dt) ** 2, dim=1))
        else:
            jerk = torch.zeros(self.num_envs, device=self.device)
        r_jerk = self._exp_shape(jerk, self.k.get('k_jerk', 0.5)) * active_f

        # Gain stability: 使用动作变化的滚动均方 (近似短窗方差)
        action_change_mag = torch.sqrt(torch.sum(action_diff**2, dim=1))
        self.action_var_accum += action_change_mag
        self.action_var_count += 1
        mean_change = self.action_var_accum / torch.clamp(self.action_var_count, min=1.0)
        # 方差近似: E[x^2] - (E[x])^2 (这里简单用 |action_change| 代替)
        var_proxy = torch.clamp(mean_change, min=0.0)
        r_gain = self._exp_shape(var_proxy, self.k.get('k_gain', 0.25)) * active_f

        # Saturation: 任一电机超过 sat_thr
        if actions.shape[1] >= 4:
            rpm_like = actions[:, :4]
            saturated = (rpm_like > self.sat_thr).any(dim=1).float()
        else:
            saturated = torch.zeros(self.num_envs, device=self.device)
        r_sat = self._exp_shape(saturated, self.k.get('k_sat', 1.0)) * active_f

        # Peak error (step-level proxy: 高误差直接惩罚)
        r_peak = self._exp_shape(pos_err, self.k.get('k_peak', 1.5)) * active_f

        # High frequency energy: 角速度差分
        hf_energy = torch.sqrt(torch.sum(omega**2, dim=1))
        r_hf = self._exp_shape(hf_energy, self.k.get('k_high_freq', 3.0)) * active_f

        # 组装加权和
        c_pos = self.w.get('position_rmse',0.0)*r_pos
        c_settle = self.w.get('settling_time',0.0)*r_settle
        c_effort = self.w.get('control_effort',0.0)*r_effort
        c_jerk = self.w.get('smoothness_jerk',0.0)*r_jerk
        c_gain = self.w.get('gain_stability',0.0)*r_gain
        c_sat = self.w.get('saturation',0.0)*r_sat
        c_peak = self.w.get('peak_error',0.0)*r_peak
        c_hf = self.w.get('high_freq',0.0)*r_hf
        total = (c_pos + c_settle + c_effort + c_jerk + c_gain + c_sat + c_peak + c_hf)
        # 累积各组件（加权后）
        self._comp_accum['position_rmse'] += c_pos
        self._comp_accum['settling_time'] += c_settle
        self._comp_accum['control_effort'] += c_effort
        self._comp_accum['smoothness_jerk'] += c_jerk
        self._comp_accum['gain_stability'] += c_gain
        self._comp_accim = self._comp_accum  # alias to avoid accidental typos
        self._comp_accum['saturation'] += c_sat
        self._comp_accum['peak_error'] += c_peak
        self._comp_accum['high_freq'] += c_hf
        return total

    def finalize(self) -> torch.Tensor:
        """在一个评估批结束时调用,用于追加峰值误差与整体下降趋势的奖励校正。"""
        # 峰值误差：误差越小惩罚越少（已经是负值）
        peak_penalty = self._exp_shape(self.max_pos_err, self.k.get('k_peak', 1.5))
        # 总下降趋势归一：下降快给正奖励
        settle_ratio = torch.clamp(self.settle_progress / (1.0 + self.steps), min=0.0, max=1.0)
        # settle_bonus: 0→1 映射为少量正奖励抵消部分惩罚
        settle_bonus = 0.5 * settle_ratio  # 最多+0.5 补偿
        # 总体仍然是负惩罚为主
        return 0.3 * peak_penalty + settle_bonus

    def get_component_totals(self) -> Dict[str, torch.Tensor]:
        """返回本批次累计的各奖励分量（加权后）的总和，shape 均为 [N]。"""
        return {k: v.clone() for k, v in self._comp_accum.items()}

    def reset_components(self, num_envs: int = None):
        """重置组件累积（用于新的一批评估）。"""
        if num_envs is None:
            num_envs = self.num_envs
        for k in self._comp_accum.keys():
            self._comp_accum[k] = torch.zeros(num_envs, device=self.device)

__all__ = ["StepwiseRewardCalculator"]
