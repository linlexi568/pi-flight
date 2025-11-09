# 控制律合成中的奖励设计理论依据

本文档概述 `reward.py` 与 `reward_stepwise.py` 所采用的奖励分量设计原则与理论依据，并给出代表性参考文献（标题/作者/年份）。

## 设计目标
- 从零合成可解释、鲁棒的控制律（而非仅优化单一轨迹误差）。
- 强调瞬态性能（settling），执行器可实现性（saturation），以及平滑性（jerk, high-frequency）。
- 采用光滑的指数形状函数 \( R = e^{-k\, m} \) 将物理量 \(m\) 归一到 \([0,1]\) 区间，保持单调性与数值稳定性。

## 奖励分量与依据

1. Position RMSE（位置均方根误差）
   - 目的：基本跟踪精度指标。
   - 依据：Karl J. Åström, Richard M. Murray, “Feedback Systems: An Introduction for Scientists and Engineers”, 2021.

2. Settling Time（稳定时间, 在 stepwise 中用“误差下降趋势”代理）
   - 目的：衡量扰动后的恢复速度与瞬态质量。
   - 依据：Katsuhiko Ogata, “Modern Control Engineering”, 2010.

3. Control Effort（控制能量, 动作差分 L2）
   - 目的：约束能耗与执行器负担；与 LQR/LQG 成本一致性。
   - 依据：Frank L. Lewis, D. Vrabie, V. Syrmos, “Optimal Control”, 2012.

4. Jerk（加加速度）
   - 目的：惩罚高 jerk，提升舒适性与降低结构疲劳；在机械臂与移动体轨迹规划中广泛使用。
   - 依据：Tamar Flash, Neville Hogan, “The Coordination of Arm Movements: An Experimentally Confirmed Mathematical Model”, Journal of Neuroscience, 1985.

5. Gain Stability（增益稳定性/抗振）
   - 目的：惩罚高频增益/动作振荡，鼓励稳健调节。
   - 依据：Sigurd Skogestad, “Simple analytic rules for model reduction and PID tuning”, 2005（以及相关鲁棒控制教材）。

6. Saturation（饱和惩罚）
   - 目的：避免执行器饱和引发的失稳与风up；工程必备约束。
   - 依据：Karl J. Åström, Tore Hägglund, “Advanced PID Control”, 2006（Anti-windup 章节）。

7. Peak Error（峰值误差）
   - 目的：最坏情况下性能/鲁棒性指标，惩罚瞬态尖峰。
   - 依据：Kemin Zhou, John C. Doyle, “Essentials of Robust Control”, 1998.

8. High-Frequency Energy（高频能量/振动）
   - 目的：降低高频激励与结构耦合风险，提升可实现性。
   - 依据：Carsten W. Scherer 等，关于频域加权与 \(H_\infty\) 设计的教材与论文（如 Robust Control 概述）。

## 形状函数与权重
- 形状函数 \( R = e^{-k m} \)：
  - 单调递减、光滑、有界；数值稳定，易于多分量加权融合。
  - \(k\) 由 `reward_profiles.py` 提供，可按任务调谐。
- 权重使用 `control_law_discovery` profile：
  - 降低纯 RMSE 的主导性，提升鲁棒性与可实现性相关项（settling, saturation, gain stability, jerk, peak）。

## 实施与工程取舍
- 在线训练使用 `reward_stepwise.py` 的逐步近似，满足实时性；
- 周期性评估与最终模型选择可用 `reward.py` 进行离线全量评估，确保科学性；
- 该“在线-离线结合”模式在工业控制优化与 AlphaZero 式搜索中常见：在线信号平滑高效，离线度量更全面。

## 参考文献（示例，不粘贴原文）
- Åström & Murray (2021) Feedback Systems
- Ogata (2010) Modern Control Engineering
- Lewis et al. (2012) Optimal Control
- Flash & Hogan (1985) Jerk-based movement model
- Skogestad (2005) PID tuning and robustness
- Åström & Hägglund (2006) Advanced PID Control (Anti-windup)
- Zhou & Doyle (1998) Essentials of Robust Control
- Scherer (2019+) Robust control and frequency-weighted design
