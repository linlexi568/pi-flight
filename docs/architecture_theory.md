# π-Flight 控制架构与理论说明（路线 A + C · v0.2）

> 版本说明：v0.2 修复了公式渲染（统一采用 `$...$` 与 `$$...$$`），并将理论叙述与当前代码模块、配置参数做了逐段对照，便于论文撰写与代码评审。

---

## 0. 系统概览：从 DSL 程序到安全动作

| 层级 | 关键模块 / 文件 | 作用 | 与理论的对应 |
| --- | --- | --- | --- |
| **程序生成** | `core/dsl.py`, `mcts_training/*` | 定义 DSL 语法、MCTS 拓展算子 | 限定程序空间 $\mathcal{P}$，并在搜索阶段保持结构约束 |
| **硬过滤** | `utils/program_constraints.py`, `BatchEvaluator.validate_program` | 通道变量白名单、规则深度限制、非法程序大惩罚 | 定义安全程序空间 $\mathcal{P}_\text{safe}$（路线 A） |
| **仿真评估** | `utils/batch_evaluation.py` + Isaac Gym | 在 2k~4k 并行环境内评估程序性能 | 提供黑箱函数 $J(p)$ 与动作统计，保证搜索过程“可控” |
| **输出安全壳** | `BatchEvaluator._apply_output_mad` | MAD（Magnitude-Angle-Delta）裁剪与速率限制 | 将所有动作投影到 $\mathcal{U}_\text{safe}$，实现 Safety by Construction |
| **表示学习** | `models/gnn_policy_nn_v2.py` | GNN 双流编码 + `get_embedding` | 为 MCTS 提供先验 policy/value 与 Ranking embedding |
| **偏序学习** | `models/ranking_value_net.py` | Pairwise Ranking + 动作特征 | 将动作统计 $[fz, tx]$ 的 mean/std/max 纳入先验（路线 C） |
| **训练入口** | `run.sh` | 写死所有超参，包含 `RANKING_GNN_CHUNK=4` | 复现实验时的外部约束，保证“不可走捷径” |

> **概念流程**：DSL 程序 → `validate_program` 筛选 → MCTS/MAD 下发到 Isaac Gym → 收集 $R(p)$、动作统计、复杂度指标 → GNN/RBVN 更新 → 形成结构化偏好，再喂回 MCTS。

---

## 1. 安全程序空间 $\mathcal{P}_\text{safe}$ 与动作安全集合 $\mathcal{U}_\text{safe}$

### 1.1 DSL 结构约束如何落地

- **程序集合**：DSL 可表达的全部程序记为 $\mathcal{P}$，每个程序 $p$ 对应控制律 $u_p: \mathcal{X} \to \mathbb{R}^4$。
- **结构谓词**：`program_constraints.validate_program` 实现的约束可抽象为
  $$C_{\text{struct}}(p) \le 0 \iff p \text{ 的所有 AST 节点都遵守通道变量白名单、算子集合与深度上界。}$$
- **安全候选集**：
  $$\mathcal{P}_{\text{struct}} = \{ p \in \mathcal{P} \mid C_{\text{struct}}(p) \le 0 \}.$$
  当前实现中，所有能进入仿真的程序都必须在此集合内。
- **通道白名单实例**：
  ```text
  u_fz : {pos_err_z, vel_z, err_i_z, err_d_z, thrust_bias, ...}
  u_tx/u_ty : {pos_err_{x,y}, err_p_{roll,pitch}, ang_vel_{x,y}, ...}
  u_tz : {pos_err_yaw, err_p_yaw, ang_vel_z, ...}
  ```
  这些集合在 `CHANNEL_VARIABLE_WHITELISTS` 中写死，确保“传感器-执行器”映射合理。

### 1.2 MAD 输出安全壳

`BatchEvaluator._apply_output_mad` 定义了**幅值 + 变化率**双重限制：

- 幅值：
  $$
  u_{fz} \in [f_{z,\min}, f_{z,\max}], \quad \lVert (u_{tx}, u_{ty}) \rVert_2 \le T_{xy,\max}, \quad |u_{tz}| \le T_{z,\max}.
  $$
- 变化率：
  $$
  |\Delta u_{fz}| \le d_{fz,\max}, \quad |\Delta u_{tx,ty,tz}| \le d_{torque,\max}.
  $$

把 MAD 壳视作投影算子 $\Pi_{\text{safe}}$：
$$
\Pi_{\text{safe}} : \mathbb{R}^4 \rightarrow \mathcal{U}_{\text{safe}}, \qquad \tilde u_p(x) = \Pi_{\text{safe}}(u_p(x)).
$$