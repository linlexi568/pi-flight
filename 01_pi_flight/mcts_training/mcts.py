import math, random, copy, hashlib
from typing import List, Any, Dict, Optional, Tuple, Set, Callable, Union

# Import DSL nodes from parent package
try:
    from ..dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode
except Exception:
    # Fallback for script mode
    import sys, pathlib
    _parent = pathlib.Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode

class MCTSNode:
    """Tree node with progressive widening & action cache.

    Attributes:
        program: Current segmented program (list of rule dicts)
        parent: Parent node
        children: List of child nodes
        visits: Visit count
        value_sum: Accumulated raw rewards (for avg value)
        depth: Depth from root (root=0)
        untried_mutations: Lazy-generated list of mutation actions (tuples)
        expanded_actions: Subset of untried already expanded (progressive widening)
    """
    def __init__(self, program: list, parent: Optional['MCTSNode']=None, depth: int=0):
        self.program = program
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.depth = depth
        self.untried_mutations: List[Tuple[str, Any]] = []
        self.expanded_actions: Set[int] = set()  # indices inside untried_mutations
        # Optional: per-child/action prior and per-node type priors (for PUCT & Dirichlet)
        self._action_priors: Dict[int, float] = {}
        self._type_priors_mixed: Optional[Dict[str, float]] = None
    @property
    def reward(self) -> float:
        return self.value_sum / self.visits if self.visits>0 else 0.0
    def is_fully_expanded(self) -> bool:
        return len(self.untried_mutations) > 0 and len(self.expanded_actions) == len(self.untried_mutations)

class MCTS_Agent:
    def __init__(self,
                 evaluation_function,
                 dsl_variables,
                 dsl_constants,
                 dsl_operators,
                 exploration_weight: float = 1.2,
                 max_depth: int = 20,
                 rollout_depth: int = 4,
                 complexity_penalty: float = 0.02,
                 pw_alpha: float = 0.6,
                 pw_c: float = 1.5,
                 transposition: bool = True,
                 warm_start_program: Optional[list] = None):
        self.evaluation_function = evaluation_function
        self.dsl_variables = dsl_variables
        self.dsl_constants = dsl_constants
        self.dsl_operators = dsl_operators
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self.complexity_penalty = complexity_penalty
        self.pw_alpha = pw_alpha
        self.pw_c = pw_c
        self.use_transposition = transposition
        # 额外 ε-greedy 探索概率（可由外部动态调度）
        self.epsilon: float = 0.0
        if warm_start_program:
            initial_program = [self._clone_rule(r) for r in warm_start_program]
        else:
            initial_program = self._generate_random_segmented_program()
        self.root = MCTSNode(initial_program, depth=0)
        self.total_iterations_done = 0
        self.last_best_reward = None
        self.best_history = []  # list of dicts
        self._global_best_program = initial_program
        self._global_best_reward = -float('inf')
        self.ttable: Dict[str, Tuple[float,int]] = {}  # hash -> (value_sum, visits)
        # 额外哈希扰动（由外部设置；用于让 TT 感知不同评估上下文，如 batch/duration）
        self._tt_salt = ""
        # --- Diversity / Novelty Tracking ---
        self.novelty_bonus_base = 0.15
        self.novelty_decay = 0.00002
        self._novelty_counter = 0
        self.mutation_stats: Dict[str,int] = {}
        self._last_improve_iter = 0
        self._stagnation_window = 120
        self._epsilon_rebound = 0.18
        self._rebound_iters = 80
        self._rebound_until_iter = 0
        # 记录程序重复出现次数（用于轻量去同化/统计）
        self._seen_counts = {}
        # 全程严格度奖励（鼓励“更窄”的条件），默认关闭，外部可注入数值（如 0.02~0.06）
        self._strict_bonus_scale = 0.0
        # 条件中允许使用的一元算子（白名单）；identity 表示不包裹
        self._allowed_cond_unaries = set(['identity', 'abs'])
        # 三角条件标准化为相位窗口 abs(trig(...)) < 小阈值 的开关与上限
        self._trig_as_phase_window = False
        self._trig_lt_max = 0.25
        # 默认复杂度调度参数（可由外部覆盖）
        self._complexity_min_scale = 0.5
        self._complexity_max_scale = 1.5
        self._complexity_ramp_start = 0.30
        self._complexity_ramp_end = 0.70
        # epsilon 调度（可由外部覆盖）：progress < end_progress 时线性衰减到 0
        self._epsilon_max = 0.25
        self._epsilon_end_progress = 0.30
        # 多样性 shaping（仅早期生效）
        self._diversity_bonus_max = 0.0  # 默认关闭
        self._diversity_end_progress = 0.30
        # --- Segmentation search bias ---
        # 保持至少 N 条规则，避免过早塌缩为单分段
        self._min_rules_guard = 2
        # 在规则数偏少时提高 add_rule 的选择概率
        self._add_rule_bias_base = 2
        # 允许复制现有规则形成新分段并微调
        self._enable_duplicate_rule = True
        # 最大规则数（可由外部覆盖）
        self._max_rules = 8
        # 可选：当出现短仿真下的新最优时，调用外部验证回调进行全长 gating
        # 签名: verify_callback(program:list, short_reward:float, iter_idx:int) -> (accepted:bool, full_reward:float|None)
        # verify 回调（由外部注入）；默认总是接受且不提供 full 分数
        # 签名: f(program:list, short_reward:float, iter_idx:int) -> (accepted:bool, full_reward:float|None)
        self.verify_callback = (lambda program, short_reward, iter_idx: (True, None))
        # 交换跨度覆盖（用于生成更多 swap_rules 组合）
        self._swap_span = 4
        # --- Macro actions & edit-type bandit credit (disabled by default) ---
        self._enable_macros = False
        # off | ucb
        self._edit_credit_mode = 'off'
        self._edit_credit_c = 0.8
        # { edit_type: { 'n': float, 'mean': float } }
        self._edit_type_stats: Dict[str, Dict[str, float]] = {}
        # --- Optional edit-type prior (e.g., NN policy prior over mutation types) ---
        # A callable taking (node, untried_actions) -> Dict[edit_type, weight]
        # returning a weight/probability per edit_type. When provided, it biases type selection in _expand.
        self._edit_prior_fn = None  # type: Optional[Callable[[MCTSNode, List[Tuple[str, Any]]], Dict[str, float]]]
        # Strength of prior shaping in type-level UCB selection. 0 disables prior influence.
        self._edit_prior_c = 0.0  # type: float
        # Optional online hook: (node, candidate_types: List[str], chosen_type: str) -> None
        self._edit_online_hook = None  # type: Optional[Callable[[MCTSNode, List[str], str], None]]

        # --- AlphaZero-lite options (default OFF) ---
        # Enable PUCT selection in _uct_select_child
        self._puct_enable = False
        self._puct_c = 1.0
        # Provide a callable returning (policy_by_type: Dict[str,float], value: float)
        # signature: f(node, available_types: List[str]) -> (Dict[type, w], value)
        self._pv_infer_fn = None  # type: Optional[Callable[[MCTSNode, List[str]], Tuple[Dict[str,float], float]]]
        # Root Dirichlet noise for exploration
        self._dirichlet_alpha = 0.3
        self._dirichlet_eps = 0.0
        # Blend factor for value head vs. environment eval
        self._value_mix_lambda = 0.0

        # --- 动态“最少规则数”调度参数（默认等于静态下限，不做下降）---
        self._min_rules_guard_initial = self._min_rules_guard
        self._min_rules_guard_final = max(1, self._min_rules_guard)  # 默认不下降
        self._min_rules_ramp_start = 0.30
        self._min_rules_ramp_end = 0.70
        self._min_rules_guard_effective = self._min_rules_guard
        # 分数几乎持平时，是否偏好分段更多的方案（允许的分数差阈值；0=关闭）
        self._prefer_more_rules_tie_delta = 0.0
        # 平分时偏好更少规则（与上者互斥使用，若都>0则先尝试 fewer 再尝试 more）
        self._prefer_fewer_rules_tie_delta = 0.0
        # 生成完整动作（P/I/D 同时设置）的概率
        self._full_action_prob = 0.0
        # 是否在 warm start 后补齐到最小规则数（默认关闭，避免破坏热启动基线）
        self._pad_after_warm_start = False

        # Pre-cache mutation actions for root
        self._ensure_mutations(self.root)
        # If warm started, evaluate once to seed best reward
        if warm_start_program:
            # warm start 若只有 1 条规则，补一条随机规则以尽早进入“分段”空间
            try:
                if bool(getattr(self, '_pad_after_warm_start', False)) and len(self.root.program) < self._min_rules_guard:
                    self.root.program.append(self._generate_random_rule())
            except Exception:
                pass
            base_val = self.evaluation_function(self.root.program)
            self._global_best_reward = base_val
            self.root.visits = 1
            self.root.value_sum = base_val
    def search(self, iterations:int, total_target: Optional[int] = None):
        for i in range(iterations):
            # 动态调度（若 total_target 提供）：前 30% 线性降 epsilon，复杂度惩罚后 50% 线性升
            if total_target:
                progress = (self.total_iterations_done + 1) / total_target
                # 暴露进度给评估使用（多样性 shaping 用）
                self._progress = progress
                # 早期：更高 epsilon 随机探索（参数化上限/结束进度）
                end_p = max(1e-6, float(getattr(self, '_epsilon_end_progress', 0.30)))
                emax = float(getattr(self, '_epsilon_max', 0.25))
                if progress < end_p:
                    self.epsilon = emax * (1 - progress / end_p)
                else:
                    self.epsilon = 0.0
                # 若处于“反弹期”，强制最低 epsilon（用于打破长期停滞）
                if self.total_iterations_done < getattr(self, '_rebound_until_iter', 0):
                    self.epsilon = max(self.epsilon, float(getattr(self, '_epsilon_rebound', 0.18)))
                # 复杂度惩罚调度：使用外部注入的 min/max scale 与 ramp 起止
                min_s = getattr(self, '_complexity_min_scale', 0.5)
                max_s = getattr(self, '_complexity_max_scale', 1.5)
                r_start = getattr(self, '_complexity_ramp_start', 0.30)
                r_end = getattr(self, '_complexity_ramp_end', 0.70)
                if progress <= r_start:
                    scale = min_s
                elif progress >= r_end:
                    scale = max_s
                else:
                    ratio = (progress - r_start) / max(1e-9, (r_end - r_start))
                    scale = min_s + (max_s - min_s) * ratio
                self._dynamic_complexity = self.complexity_penalty * scale
                # 动态最少规则下限：随进度从 initial -> final 线性下降
                g0 = getattr(self, '_min_rules_guard_initial', self._min_rules_guard)
                g1 = getattr(self, '_min_rules_guard_final', self._min_rules_guard)
                gs = getattr(self, '_min_rules_ramp_start', 0.30)
                ge = getattr(self, '_min_rules_ramp_end', 0.70)
                if progress <= gs:
                    g_eff = g0
                elif progress >= ge:
                    g_eff = g1
                else:
                    ratio = (progress - gs) / max(1e-9, (ge - gs))
                    g_eff = int(round(g0 + (g1 - g0) * ratio))
                self._min_rules_guard_effective = max(1, int(g_eff))
            node=self._select(self.root)
            reward=self._evaluate_node(node)
            # Bandit: record immediate delta for the edit type applied to reach this node
            try:
                act = getattr(node, '_applied_action', None)
                if act is not None and isinstance(act, tuple) and len(act) >= 1:
                    base = getattr(node, '_applied_parent_reward', None)
                    if base is None and node.parent is not None and node.parent.visits > 0:
                        base = node.parent.reward
                    base_val = float(base) if isinstance(base, (int, float)) else 0.0
                    if getattr(self, '_edit_credit_mode', 'off') == 'ucb':
                        self._update_edit_credit(str(act[0]), float(reward) - base_val)
            except Exception:
                pass
            self._backpropagate(node,reward)
            self.total_iterations_done+=1
            # 更新全局最佳（使用节点 reward 而不是 root 均值）
            if reward > self._global_best_reward:
                accepted = True
                accepted_reward = reward
                _vcb = getattr(self, 'verify_callback', None)
                if _vcb is not None and callable(_vcb):
                    try:
                        _res = _vcb(node.program, reward, self.total_iterations_done + 1)
                        _acpt = False
                        _full_r = None
                        if isinstance(_res, tuple):
                            if len(_res) >= 1:
                                _acpt = bool(_res[0])
                            if len(_res) >= 2 and isinstance(_res[1], (int, float)):
                                _full_r = float(_res[1])
                        else:
                            _acpt = bool(_res)
                        accepted = bool(_acpt)
                        if accepted and _full_r is not None:
                            accepted_reward = _full_r
                    except Exception:
                        # 回调失败时回退为直接接受短仿真分数
                        accepted = True
                        accepted_reward = reward
                if accepted:
                    self._global_best_reward = accepted_reward
                    self._global_best_program = [self._clone_rule(r) for r in node.program]
                    # 记录最近一次有效提升的迭代（用于停滞检测）
                    self._last_improve_iter = self.total_iterations_done
            else:
                # 平分偏好（两种策略，少/多规则，少规则优先）：
                prefer_fewer = float(getattr(self, '_prefer_fewer_rules_tie_delta', 0.0))
                prefer_more = float(getattr(self, '_prefer_more_rules_tie_delta', 0.0))
                # 1) 偏好更少规则（若配置）：reward + delta >= best 且 candidate 规则更少
                applied_tie=False
                if prefer_fewer > 0.0:
                    try:
                        fewer_rules = len(node.program) < len(self._global_best_program)
                    except Exception:
                        fewer_rules = False
                    if fewer_rules and (reward + prefer_fewer) >= self._global_best_reward:
                        applied_tie=True
                # 2) 否则，偏好更多规则（若配置）：reward + delta >= best 且 candidate 规则更多
                if (not applied_tie) and prefer_more > 0.0:
                    try:
                        more_rules = len(node.program) > len(self._global_best_program)
                    except Exception:
                        more_rules = False
                    if more_rules and (reward + prefer_more) >= self._global_best_reward:
                        accepted = True
                        accepted_reward = reward
                        _vcb = getattr(self, 'verify_callback', None)
                        if _vcb is not None and callable(_vcb):
                            try:
                                _res = _vcb(node.program, reward, self.total_iterations_done + 1)
                                _acpt = False
                                _full_r = None
                                if isinstance(_res, tuple):
                                    if len(_res) >= 1:
                                        _acpt = bool(_res[0])
                                    if len(_res) >= 2 and isinstance(_res[1], (int, float)):
                                        _full_r = float(_res[1])
                                else:
                                    _acpt = bool(_res)
                                accepted = bool(_acpt)
                                if accepted and _full_r is not None:
                                    accepted_reward = _full_r
                            except Exception:
                                accepted = True
                                accepted_reward = reward
                        if accepted:
                            # 仅替换程序，保持 best_reward 不下降（可选：若 full 优于当前可同步提升）
                            self._global_best_program = [self._clone_rule(r) for r in node.program]
                            if accepted_reward > self._global_best_reward:
                                self._global_best_reward = accepted_reward
                            self._last_improve_iter = self.total_iterations_done
            self.last_best_reward=self._global_best_reward
            rules_count=len(self._global_best_program)
            self.best_history.append({'iter': self.total_iterations_done, 'reward': float(self._global_best_reward), 'rules': rules_count})
            # 若长期未提升，则触发一段“反弹探索期”：抬高 epsilon
            try:
                stagn_win = int(getattr(self, '_stagnation_window', 0) or 0)
                if stagn_win > 0:
                    since = self.total_iterations_done - int(getattr(self, '_last_improve_iter', 0) or 0)
                    if since >= stagn_win and self.total_iterations_done >= int(getattr(self, '_rebound_until_iter', 0) or 0):
                        self._rebound_until_iter = self.total_iterations_done + int(getattr(self, '_rebound_iters', 80))
                        # 进入反弹期：清空转置表与重复计数，避免旧估计继续束缚搜索
                        try:
                            self.ttable.clear()
                            self._seen_counts.clear()
                        except Exception:
                            pass
            except Exception:
                pass
            if total_target is None:
                print(f"迭代 {i+1}/{iterations} | 当前最佳奖励: {self._global_best_reward:.4f}", end='\r')
            else:
                print(f"迭代 {self.total_iterations_done}/{total_target} | 当前最佳奖励: {self._global_best_reward:.4f}", end='\r')
        print("\n搜索完成。")
    def _select(self,node:MCTSNode)->MCTSNode:
        # Selection with progressive widening until leaf or depth limit
        while True:
            if node.depth >= self.max_depth:
                return node
            self._ensure_mutations(node)
            # Progressive widening condition
            max_children = int(self.pw_c * (node.visits ** self.pw_alpha)) if node.visits>0 else 1
            can_expand = len(node.expanded_actions) < len(node.untried_mutations) and len(node.children) < max_children
            if can_expand:
                return self._expand(node)
            if not node.children:
                return node
            node = self._uct_select_child(node)
        return node
    def _uct_select_child(self,node:MCTSNode)->MCTSNode:
        # If PUCT disabled, fall back to classic UCB1
        if not getattr(self, '_puct_enable', False):
            log_total_visits = math.log(node.visits) if node.visits>0 else 0
            best_child=None; best_uct=-float('inf')
            # epsilon-greedy: 随机探索子节点
            if getattr(self, 'epsilon', 0.0) > 0 and random.random() < self.epsilon:
                return random.choice(node.children)
            for child in node.children:
                if child.visits==0:
                    return child
                mean_value = child.reward
                uct = mean_value + self.exploration_weight * math.sqrt(log_total_visits/child.visits)
                if uct>best_uct:
                    best_uct=uct; best_child=child
            return best_child if best_child is not None else random.choice(node.children)
        # PUCT selection
        total_n = sum(max(0, c.visits) for c in node.children)
        sqrt_total = math.sqrt(max(1.0, float(total_n)))
        c_puct = float(getattr(self, '_puct_c', 1.0) or 1.0)
        best=None; best_score=-float('inf')
        # epsilon-greedy fallback
        if getattr(self, 'epsilon', 0.0) > 0 and random.random() < self.epsilon:
            return random.choice(node.children)
        for idx, child in enumerate(node.children):
            q = float(child.reward) if child.visits>0 else 0.0
            p = float(getattr(child, '_prior_p', 0.0) or 0.0)
            u = c_puct * p * (sqrt_total / (1.0 + float(child.visits)))
            score = q + u
            if score > best_score:
                best_score = score; best = child
        return best if best is not None else random.choice(node.children)
    def _expand(self,node:MCTSNode)->MCTSNode:
        self._ensure_mutations(node)
        # Choose first unexpanded action
        unexpanded_indices = [i for i in range(len(node.untried_mutations)) if i not in node.expanded_actions]
        if not unexpanded_indices:
            return node  # nothing to expand
        # Type-level UCB selection if enabled
        idx = None
        if getattr(self, '_edit_credit_mode', 'off') == 'ucb':
            type_to_indices: Dict[str, list] = {}
            for i in unexpanded_indices:
                etype = node.untried_mutations[i][0]
                type_to_indices.setdefault(etype, []).append(i)
            total_n = 1 + sum(int(self._edit_type_stats.get(t, {}).get('n', 0)) for t in type_to_indices.keys())
            best_t = None; best_score = -float('inf')
            # Optional prior over edit types
            priors = {}
            try:
                fn = getattr(self, '_edit_prior_fn', None)
                if fn is not None and callable(fn):
                    _res = fn(node, [node.untried_mutations[i] for i in unexpanded_indices])
                    if isinstance(_res, dict):
                        priors = {str(k): float(v) for k, v in _res.items() if v is not None}
            except Exception:
                priors = {}
            c_prior = float(getattr(self, '_edit_prior_c', 0.0) or 0.0)
            for t in type_to_indices.keys():
                st = self._edit_type_stats.get(t, {'n': 0.0, 'mean': 0.0})
                n = float(st.get('n', 0.0) or 0.0)
                mean = float(st.get('mean', 0.0) or 0.0)
                c = float(getattr(self, '_edit_credit_c', 0.8) or 0.8)
                bonus = c * math.sqrt(math.log(max(2.0, float(total_n))) / (n + 1.0))
                prior_term = 0.0
                try:
                    # Use prior weight if provided; apply small epsilon smoothing
                    p = float(priors.get(t, 0.0) or 0.0)
                    if p > 0.0 and c_prior > 0.0:
                        prior_term = c_prior * p
                except Exception:
                    prior_term = 0.0
                score = mean + bonus + prior_term
                if score > best_score:
                    best_score = score; best_t = t
            if best_t is not None:
                cand = type_to_indices.get(best_t, [])
                if cand:
                    idx = random.choice(cand)
        if idx is None:
            # If no UCB bandit or selection failed, optionally sample by prior over types
            try:
                fn = getattr(self, '_edit_prior_fn', None)
                if fn is not None and callable(fn):
                    # Build mapping type -> unexpanded indices
                    type_to_indices = {}
                    for i in unexpanded_indices:
                        etype = node.untried_mutations[i][0]
                        type_to_indices.setdefault(etype, []).append(i)
                    _res = fn(node, [node.untried_mutations[i] for i in unexpanded_indices])
                    priors = {str(k): float(v) for k, v in _res.items()} if isinstance(_res, dict) else {}
                    # Form a candidate list weighted by priors per type
                    weighted = []
                    for t, inds in type_to_indices.items():
                        w = float(priors.get(t, 0.0) or 0.0)
                        w = max(0.0, w)
                        if w == 0.0:
                            # Ensure every type retains a chance
                            w = 1e-6
                        # Distribute weight uniformly among indices of that type
                        w_each = w / max(1, len(inds))
                        for ii in inds:
                            weighted.append((ii, w_each))
                    if weighted:
                        # Sample proportionally
                        s = sum(w for _, w in weighted)
                        r = random.random() * s
                        cum = 0.0
                        chosen = weighted[0][0]
                        for ii, w in weighted:
                            cum += w
                            if r <= cum:
                                chosen = ii; break
                        idx = chosen
            except Exception:
                idx = None
            if idx is None:
                idx = random.choice(unexpanded_indices)
        action = node.untried_mutations[idx]
        # online training hook (collect sample)
        try:
            hook = getattr(self, '_edit_online_hook', None)
            if hook is not None and callable(hook):
                # candidate types at this decision point
                cand_types = []
                try:
                    for i in unexpanded_indices:
                        et = node.untried_mutations[i][0]
                        if isinstance(et, str):
                            cand_types.append(et)
                    # deduplicate preserve order
                    seen = set(); tmp = []
                    for t in cand_types:
                        if t not in seen:
                            seen.add(t); tmp.append(t)
                    cand_types = tmp
                except Exception:
                    cand_types = []
                chosen_t = action[0] if isinstance(action, tuple) and len(action) > 0 and isinstance(action[0], str) else None
                if isinstance(chosen_t, str):
                    hook(node, cand_types, chosen_t)
        except Exception:
            pass
        # Compute and cache per-action prior for PUCT
        try:
            # Build available type set
            avail_types = []
            for i in unexpanded_indices:
                et = node.untried_mutations[i][0]
                if isinstance(et, str):
                    avail_types.append(et)
            # dedup preserve order
            seen = set(); tmp = []
            for t in avail_types:
                if t not in seen:
                    seen.add(t); tmp.append(t)
            avail_types = tmp
            type_priors = {}
            # Prefer PV infer fn if provided; else fall back to edit_prior_fn; else uniform
            pv_fn = getattr(self, '_pv_infer_fn', None)
            if pv_fn is not None and callable(pv_fn):
                try:
                    _res = pv_fn(node, avail_types)
                    if isinstance(_res, tuple):
                        _pmap = _res[0] if len(_res) > 0 else None
                    else:
                        _pmap = _res
                    if isinstance(_pmap, dict):
                        type_priors = {str(k): float(v) for k, v in _pmap.items() if v is not None}
                    else:
                        type_priors = {}
                except Exception:
                    type_priors = {}
            if not type_priors:
                try:
                    fn = getattr(self, '_edit_prior_fn', None)
                    if fn is not None and callable(fn):
                        _res = fn(node, [node.untried_mutations[i] for i in unexpanded_indices])
                        if isinstance(_res, dict):
                            type_priors = {str(k): float(v) for k, v in _res.items() if v is not None}
                except Exception:
                    type_priors = {}
            if not type_priors and avail_types:
                u = 1.0 / float(len(avail_types))
                type_priors = {t: u for t in avail_types}
            # Root Dirichlet noise (mix once per node)
            try:
                if node.depth == 0 and float(getattr(self, '_dirichlet_eps', 0.0) or 0.0) > 0.0:
                    if node._type_priors_mixed is None:
                        eps = float(getattr(self, '_dirichlet_eps', 0.25) or 0.25)
                        alpha = float(getattr(self, '_dirichlet_alpha', 0.3) or 0.3)
                        import numpy as _np
                        if avail_types:
                            noise = _np.random.gamma(alpha, 1.0, size=len(avail_types))
                            noise = noise / max(1e-12, noise.sum())
                            base = _np.array([float(type_priors.get(t, 0.0) or 0.0) for t in avail_types], dtype=_np.float64)
                            # normalize base
                            bsum = float(base.sum())
                            if bsum > 0:
                                base = base / bsum
                            mixed = (1.0 - eps) * base + eps * noise
                            node._type_priors_mixed = {t: float(mixed[i]) for i, t in enumerate(avail_types)}
                    if isinstance(node._type_priors_mixed, dict) and node._type_priors_mixed:
                        type_priors = node._type_priors_mixed
            except Exception:
                pass
            # Assign action prior from its type
            a_type = action[0] if isinstance(action, tuple) and len(action) > 0 else None
            if isinstance(a_type, str):
                p = float(type_priors.get(a_type, 0.0) or 0.0)
            else:
                p = 0.0
            # cache by action index
            try:
                node._action_priors[idx] = p
            except Exception:
                pass
        except Exception:
            pass

        new_program = self._apply_mutation(copy.deepcopy(node.program), action)
        child = MCTSNode(new_program, parent=node, depth=node.depth+1)
        # Store applied action and parent reward for bandit delta
        try:
            setattr(child, '_applied_action', action)
            setattr(child, '_applied_parent_reward', float(node.reward) if node.visits>0 else 0.0)
            # propagate prior p for this edge to child for PUCT
            if idx in getattr(node, '_action_priors', {}):
                setattr(child, '_prior_p', float(node._action_priors.get(idx, 0.0)))
        except Exception:
            pass
        node.children.append(child)
        node.expanded_actions.add(idx)
        # Pre-cache for child
        self._ensure_mutations(child)
        return child
    def _evaluate_node(self, node: MCTSNode) -> float:
        # Transposition lookup
        h = self._hash_program(node.program)
        # 记录出现次数（轻量新颖度统计）
        try:
            self._seen_counts[h] = self._seen_counts.get(h, 0) + 1
        except Exception:
            pass
        if self.use_transposition and h in self.ttable:
            val_sum, visits = self.ttable[h]
            # reuse mean value but treat as a light evaluation; still add penalty for complexity
            base_val = val_sum/visits
            # 累积访问次数（便于后续基于访问频次做轻度惩罚/引导）
            try:
                self.ttable[h] = (val_sum, visits + 1)
            except Exception:
                pass
        else:
            # Optional value head mix to reduce expensive eval
            v_est = None
            lam = float(getattr(self, '_value_mix_lambda', 0.0) or 0.0)
            if lam > 0.0:
                try:
                    # Build available types for current node
                    self._ensure_mutations(node)
                    avail_types = []
                    for i in range(len(node.untried_mutations)):
                        et = node.untried_mutations[i][0]
                        if isinstance(et, str):
                            avail_types.append(et)
                    # dedup
                    seen = set(); tmp = []
                    for t in avail_types:
                        if t not in seen:
                            seen.add(t); tmp.append(t)
                    avail_types = tmp
                    pv_fn = getattr(self, '_pv_infer_fn', None)
                    if pv_fn is not None and callable(pv_fn):
                        _res = pv_fn(node, avail_types)
                        # Accept (dict, value) or just value
                        if isinstance(_res, tuple):
                            v_part = _res[1] if len(_res) > 1 else None
                        else:
                            v_part = _res
                        if isinstance(v_part, (int, float)):
                            v_est = float(v_part)
                except Exception:
                    v_est = None
            # Base environment eval
            try:
                env_val = self.evaluation_function(node.program)
            except Exception:
                env_val = 0.0
            if v_est is not None and lam > 0.0:
                base_val = float((1.0 - lam) * float(env_val) + lam * float(v_est))
            else:
                base_val = float(env_val)
            if self.use_transposition:
                self.ttable[h] = (base_val, 1)
        # Optional rollout playout for leaf nodes (depth-limited)
        rollout_bonus = 0.0
        if node.depth < self.max_depth and (not node.children):
            rollout_bonus = self._rollout(node.program, self.rollout_depth)
        complexity = len(node.program)
        # 使用动态复杂度（如有）
        cpen = getattr(self, '_dynamic_complexity', self.complexity_penalty)
        penalized = base_val - cpen * (complexity-1)
        # 对重复出现过多的程序给出极轻的惩罚（推动跳出局部最优），默认极小
        try:
            seen = float(self._seen_counts.get(h, 0))
            # 仅当 seen 超过 5 次后才开始产生可见影响；系数很小，避免过度干预
            repeat_pen = max(0.0, (seen - 5.0)) * float(getattr(self, 'novelty_decay', self.novelty_decay))
            penalized -= repeat_pen
        except Exception:
            pass
    # 早期多样性 shaping：鼓励更“窄”的条件与更均衡的变量使用
        div_bonus = 0.0
        p = float(getattr(self, '_progress', 1.0))
        div_max = float(getattr(self, '_diversity_bonus_max', 0.0))
        if div_max > 0.0 and p < float(getattr(self, '_diversity_end_progress', 0.30)):
            # 1) 条件“窄度”：统计比较阈值的绝对值是否偏小（对 '<' 用小阈值、对 '>' 用较大阈值都视为窄）
            narrow_score = self._estimate_narrowness(node.program)
            # 2) 变量多样性：不同变量的覆盖越广，得分越高
            var_div = self._estimate_variable_diversity(node.program)
            # 线性随时间衰减
            decay = 1.0 - (p / max(1e-6, float(getattr(self, '_diversity_end_progress', 0.30))))
            div_bonus = div_max * (0.6 * narrow_score + 0.4 * var_div) * max(0.0, decay)
        strict_bonus = float(getattr(self, '_strict_bonus_scale', 0.0)) * self._estimate_strictness(node.program)
        return penalized + 0.1 * rollout_bonus + div_bonus + strict_bonus
    def _rollout(self, program: list, depth: int) -> float:
        cur = copy.deepcopy(program)
        best = -float('inf')
        for _ in range(depth):
            action = self._sample_random_mutation(cur)
            cur = self._apply_mutation(cur, action)
            val = self.evaluation_function(cur)
            if val>best:
                best=val
        return best if best>-float('inf') else 0.0
    def _backpropagate(self,node: Optional[MCTSNode],reward:float):
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent

    # --- Bandit: update edit-type credit with delta improvement ---
    def _update_edit_credit(self, edit_type: str, delta: float):
        try:
            st = self._edit_type_stats.get(edit_type, {'n': 0.0, 'mean': 0.0})
            n = float(st.get('n', 0.0) or 0.0)
            m = float(st.get('mean', 0.0) or 0.0)
            n_new = n + 1.0
            m_new = m + (float(delta) - m) / n_new
            self._edit_type_stats[edit_type] = {'n': n_new, 'mean': m_new}
        except Exception:
            pass
    def get_best_program(self):
        return self._global_best_program, self._global_best_reward
    def set_verify_callback(self, cb: Optional[Union[Callable[[list, float, int], Tuple[bool, Optional[float]]], Callable[..., tuple]]]):
        """Set gating verify callback; any callable that returns (accepted:bool, full_reward:Optional[float]) is OK."""
        if cb is None:
            self.verify_callback = (lambda program, short_reward, iter_idx: (True, None))
        else:
            self.verify_callback = cb  # type: ignore

    def inject_candidate(self, program: list, assumed_reward: Optional[float] = None, iter_idx: Optional[int] = None) -> Tuple[bool, float]:
        """Externally inject a candidate program as a potential new best.

        - program: segmented program (list of rules)
        - assumed_reward: if provided, treat it as the short-eval score used for gating; else evaluate.
        - iter_idx: logical iteration index for logging/gating; if None, uses total_iterations_done+1.

        Returns: (accepted, best_reward_after)
        """
        try:
            short_r = float(assumed_reward) if isinstance(assumed_reward, (int, float)) else float(self.evaluation_function(program))
        except Exception:
            # evaluation failed, reject
            return (False, float(self._global_best_reward))
        accepted = True
        accepted_reward = short_r
        _vcb = getattr(self, 'verify_callback', None)
        it = int(iter_idx) if isinstance(iter_idx, int) else int(self.total_iterations_done + 1)
        if _vcb is not None and callable(_vcb):
            try:
                _res = _vcb(program, short_r, it)
                _acpt = False
                _full_r = None
                if isinstance(_res, tuple):
                    if len(_res) >= 1:
                        _acpt = bool(_res[0])
                    if len(_res) >= 2 and isinstance(_res[1], (int, float)):
                        _full_r = float(_res[1])
                else:
                    _acpt = bool(_res)
                accepted = bool(_acpt)
                if accepted and _full_r is not None:
                    accepted_reward = _full_r
            except Exception:
                accepted = True
                accepted_reward = short_r
        if accepted:
            # Only update if not worse to avoid regressions when gating returns a small drop
            if accepted_reward >= self._global_best_reward:
                self._global_best_program = [self._clone_rule(r) for r in program]
                self._global_best_reward = float(accepted_reward)
                self._last_improve_iter = int(self.total_iterations_done)
                # record into history with current logical iter
                try:
                    rules_count = len(self._global_best_program)
                    self.best_history.append({'iter': int(self.total_iterations_done), 'reward': float(self._global_best_reward), 'rules': rules_count})
                except Exception:
                    pass
            return (True, float(self._global_best_reward))
        return (False, float(self._global_best_reward))
    def program_to_str(self, program:list)->str:
        if not isinstance(program,list): return str(program)
        rule_strings=[]
        for i,rule in enumerate(program):
            condition_str=self._ast_to_str(rule['condition'])
            action_parts=[]
            for act in rule['action']:
                if isinstance(act,BinaryOpNode) and act.op=='set' and isinstance(act.left,TerminalNode):
                    rstr = self._ast_to_str(act.right) if hasattr(act, 'right') else '0'
                    action_parts.append(f"{act.left.value} = {rstr}")
            action_str=", ".join(action_parts)
            rule_strings.append(f"  Rule {i}: IF ({condition_str}) THEN ({action_str})")
        return "\n"+"\n".join(rule_strings)
    # --- Mutation action system ---
    def _ensure_mutations(self, node: MCTSNode):
        if node.untried_mutations:
            return
        actions = []
        # Pre-enumerate possible mutations (bounded) without stochastic params yet (param mutations will sample on apply)
        # add_rule: 根据当前规则数添加多次以提高被选概率
        add_bias = self._add_rule_bias_base + max(0, getattr(self, '_min_rules_guard_effective', self._min_rules_guard) - len(node.program))
        for _ in range(max(1, add_bias)):
            actions.append(('add_rule', None))
        if node.program:
            for idx in range(len(node.program)):
                actions.append(('remove_rule', idx))
                actions.append(('mutate_condition', idx))
                actions.append(('mutate_action', idx))
                actions.append(('tweak_multiplier', idx))
                # 细粒度抛光：更小步幅的增益与阈值微调
                actions.append(('micro_tweak', idx))
                actions.append(('nudge_threshold', idx))
                actions.append(('narrow_condition', idx))
                actions.append(('promote_rule', idx))
                actions.append(('split_rule', idx))
                # Macro actions (optional)
                if getattr(self, '_enable_macros', False):
                    actions.append(('macro_triplet_tune', idx))
                    actions.append(('macro_refine_condition', idx))
                if self._enable_duplicate_rule and len(node.program) < getattr(self, '_max_rules', 8):
                    actions.append(('duplicate_rule', idx))
            if len(node.program) > 1:
                # 扩大 swap 覆盖范围（可配置跨度）
                indices = list(range(len(node.program)))
                random.shuffle(indices)
                span = int(getattr(self, '_swap_span', 4))
                k_pairs = min(span, len(indices)-1)
                for i in range(k_pairs):
                    j_lim = min(i + 1 + span, len(indices))
                    for j in range(i+1, j_lim):
                        actions.append(('swap_rules', (indices[i], indices[j])))
        node.untried_mutations = actions
    def _sample_random_mutation(self, program: list) -> Tuple[str, Any]:
        mock_node = MCTSNode(program, None, 0)
        self._ensure_mutations(mock_node)
        return random.choice(mock_node.untried_mutations)
    def _apply_mutation(self, program: list, action: Tuple[str, Any]) -> list:
        mutation_type, payload = action
        MAX_RULES = getattr(self, '_max_rules', 8)
        if mutation_type == 'add_rule' and len(program) < MAX_RULES:
            program.append(self._generate_random_rule())
        elif mutation_type == 'remove_rule':
            idx = payload
            guard_limit = max(1, getattr(self, '_min_rules_guard_effective', getattr(self, '_min_rules_guard', 1)))
            if len(program) > guard_limit and 0 <= idx < len(program):
                program.pop(idx)
        elif mutation_type == 'mutate_condition':
            idx = payload
            if 0 <= idx < len(program):
                c_new = self._generate_random_condition()
                program[idx]['condition'] = self._enforce_narrow_condition(c_new)
        elif mutation_type == 'mutate_action':
            idx = payload
            if 0 <= idx < len(program):
                program[idx]['action'] = self._generate_random_action()
        elif mutation_type == 'narrow_condition':
            idx = payload
            if 0 <= idx < len(program):
                cond = program[idx]['condition']
                if isinstance(cond, BinaryOpNode) and cond.op in ('<','>') and isinstance(cond.right, TerminalNode) and isinstance(cond.right.value, (int,float)):
                    T = float(cond.right.value)
                    if cond.op == '<':
                        if T >= 0:
                            T_new = T * random.uniform(0.75, 0.92)  # 收紧: 原 0.5-0.85
                        else:
                            T_new = T * random.uniform(0.90, 1.0)
                    else:
                        if T >= 0:
                            T_new = T * random.uniform(1.08, 1.30)  # 收紧: 原 1.15-1.6
                        else:
                            T_new = T * random.uniform(0.75, 0.92)
                    cond.right = TerminalNode(round(float(T_new), 4))
                    program[idx]['condition'] = self._enforce_narrow_condition(cond)
        elif mutation_type == 'tweak_multiplier':
            # 泛化为：微调动作表达式中的常数项
            idx = payload
            if 0 <= idx < len(program):
                acts = program[idx]['action']
                const_nodes = []
                def collect_consts(n: ProgramNode):
                    if isinstance(n, TerminalNode) and isinstance(n.value, (int, float)):
                        const_nodes.append(n)
                    elif isinstance(n, UnaryOpNode):
                        collect_consts(n.child)
                    elif isinstance(n, BinaryOpNode):
                        collect_consts(n.left); collect_consts(n.right)
                for a in acts:
                    if isinstance(a, BinaryOpNode) and a.op == 'set' and hasattr(a, 'right'):
                        collect_consts(a.right)
                if const_nodes:
                    n = random.choice(const_nodes)
                    try:
                        noise = random.uniform(0.92, 1.08)
                        n.value = round(float(n.value) * noise, 4)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        elif mutation_type == 'micro_tweak':
            # 更小步幅的常数微调
            idx = payload
            if 0 <= idx < len(program):
                acts = program[idx]['action']
                const_nodes = []
                def collect_consts(n: ProgramNode):
                    if isinstance(n, TerminalNode) and isinstance(n.value, (int, float)):
                        const_nodes.append(n)
                    elif isinstance(n, UnaryOpNode):
                        collect_consts(n.child)
                    elif isinstance(n, BinaryOpNode):
                        collect_consts(n.left); collect_consts(n.right)
                for a in acts:
                    if isinstance(a, BinaryOpNode) and a.op == 'set' and hasattr(a, 'right'):
                        collect_consts(a.right)
                if const_nodes:
                    n = random.choice(const_nodes)
                    try:
                        noise = random.uniform(0.97, 1.03)
                        n.value = round(float(n.value) * noise, 4)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        elif mutation_type == 'nudge_threshold':
            # 条件阈值的微幅调整，便于在边缘处找到增益
            idx = payload
            if 0 <= idx < len(program):
                cond = program[idx]['condition']
                if isinstance(cond, BinaryOpNode) and cond.op in ('<','>') and isinstance(cond.right, TerminalNode) and isinstance(cond.right.value,(int,float)):
                    T = float(cond.right.value)
                    # 微小比例因子；保持正负号不变
                    factor = random.uniform(0.97, 1.03)
                    T_new = T * factor
                    cond.right = TerminalNode(round(T_new, 4))
                    program[idx]['condition'] = self._enforce_narrow_condition(cond)
        elif mutation_type == 'duplicate_rule':
            idx = payload
            if 0 <= idx < len(program) and len(program) < MAX_RULES:
                base = program[idx]
                new_rule = {'condition': base['condition'], 'action': [a for a in base['action']]}
                if random.random() < 0.6:
                    acts = new_rule['action']
                    # 复制后对表达式中的常数做轻微抖动
                    const_nodes = []
                    def collect_consts(n: ProgramNode):
                        if isinstance(n, TerminalNode) and isinstance(n.value, (int, float)):
                            const_nodes.append(n)
                        elif isinstance(n, UnaryOpNode):
                            collect_consts(n.child)
                        elif isinstance(n, BinaryOpNode):
                            collect_consts(n.left); collect_consts(n.right)
                    for a in acts:
                        if isinstance(a, BinaryOpNode) and a.op == 'set' and hasattr(a, 'right'):
                            collect_consts(a.right)
                    for cn in const_nodes:
                        try:
                            cn.value = round(float(cn.value) * random.uniform(0.9, 1.15), 4)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                else:
                    new_rule['condition'] = self._generate_random_condition()
                program.append(new_rule)
        elif mutation_type == 'promote_rule':
            idx = payload
            if 0 <= idx < len(program):
                step = random.randint(1, max(1, idx))
                new_pos = max(0, idx - step)
                rule = program.pop(idx)
                program.insert(new_pos, rule)
        elif mutation_type == 'split_rule':
            idx = payload
            if 0 <= idx < len(program):
                cond = program[idx].get('condition')
                acts = [a for a in program[idx].get('action', [])]
                if isinstance(cond, BinaryOpNode) and cond.op in ('<','>') and isinstance(cond.right, TerminalNode) and isinstance(cond.right.value,(int,float)):
                    T = float(cond.right.value)
                    if cond.op == '<':
                        f1, f2 = random.uniform(0.5, 0.75), random.uniform(0.75, 0.9)
                        c1 = BinaryOpNode('<', cond.left, TerminalNode(round(T * f1, 4)))
                        c2 = BinaryOpNode('<', cond.left, TerminalNode(round(T * f2, 4)))
                    else:
                        base = max(T, 0.05)
                        f1, f2 = random.uniform(1.2, 1.8), random.uniform(1.8, 2.6)
                        c1 = BinaryOpNode('>', cond.left, TerminalNode(round(base * f1, 4)))
                        c2 = BinaryOpNode('>', cond.left, TerminalNode(round(base * f2, 4)))
                    c1 = self._enforce_narrow_condition(c1)
                    c2 = self._enforce_narrow_condition(c2)
                    def _jitter(asrc:list):
                        outs = [a for a in asrc]
                        mult_nodes=[a for a in outs if isinstance(a,BinaryOpNode) and a.op=='set']
                        for mn in mult_nodes:
                            if isinstance(mn.right, TerminalNode) and isinstance(mn.right.value,(int,float)):
                                mn.right = TerminalNode(round(float(mn.right.value)*random.uniform(0.9,1.15),4))
                        return outs
                    r1 = {'condition': c1, 'action': _jitter(acts)}
                    r2 = {'condition': c2, 'action': _jitter(acts)}
                    program.pop(idx)
                    if len(program) < MAX_RULES:
                        program.insert(idx, r2)
                    if len(program) < MAX_RULES:
                        program.insert(idx, r1)
        elif mutation_type == 'swap_rules':
            i1,i2 = payload
            if 0 <= i1 < len(program) and 0 <= i2 < len(program) and i1!=i2:
                program[i1], program[i2] = program[i2], program[i1]
        # --- Macro actions ---
        elif mutation_type == 'macro_triplet_tune':
            # 数学原语模式：整体缩放该规则中动作表达式的常数
            idx = payload
            if 0 <= idx < len(program):
                acts = program[idx].get('action', [])
                scale = random.uniform(0.95, 1.05)
                def scale_consts(n: ProgramNode):
                    if isinstance(n, TerminalNode) and isinstance(n.value, (int, float)):
                        try:
                            n.value = round(float(n.value) * scale, 4)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    elif isinstance(n, UnaryOpNode):
                        scale_consts(n.child)
                    elif isinstance(n, BinaryOpNode):
                        scale_consts(n.left); scale_consts(n.right)
                for a in acts:
                    if isinstance(a, BinaryOpNode) and a.op == 'set' and hasattr(a, 'right'):
                        scale_consts(a.right)
        elif mutation_type == 'macro_refine_condition':
            idx = payload
            if 0 <= idx < len(program):
                # compound: narrow then nudge
                program = self._apply_mutation(program, ('narrow_condition', idx))
                program = self._apply_mutation(program, ('nudge_threshold', idx))
        return program
    def _generate_random_rule(self)->Dict[str,Any]:
        return {'condition': self._generate_random_condition(), 'action': self._generate_random_action()}
    def _generate_random_segmented_program(self,num_rules=2)->list:
        return [self._generate_random_rule() for _ in range(num_rules)]
    def _generate_random_condition(self,depth=0,max_depth=3)->ProgramNode:
        import random, math
        comparators = ['>', '<']
        # 主运算: trig + tan + 幅度压缩函数
        unary_ops_primary = ['sin', 'cos', 'tan', 'log1p', 'sqrt']
        # 次级包裹：符号 / 绝对值（平滑或折叠）
        unary_ops_secondary = ['abs', 'sign']

        def maybe_unary(node: ProgramNode, p_primary: float, p_secondary: float, allow_second_layer: bool = True):
            """随机包裹一元函数；primary 为 trig/log1p/sqrt/tan，secondary 为 abs/sign。
            allow_second_layer 控制是否允许在 primary 外再包一层 abs 以平滑极值。"""
            allowed = getattr(self, '_allowed_cond_unaries', set(['identity','abs']))
            # primary 集合与白名单的交集
            prim_pool = [op for op in unary_ops_primary if op in allowed]
            sec_pool = [op for op in unary_ops_secondary if op in allowed]
            if prim_pool and random.random() < p_primary:  # 触发 primary 类且已允许
                op = random.choice(prim_pool)
                node = UnaryOpNode(op, node)
                if allow_second_layer and sec_pool and random.random() < p_secondary:  # 少量再包 abs/sign
                    node = UnaryOpNode(random.choice(sec_pool), node)
                return node
            # 未触发 primary，则小概率只包 abs/sign
            if sec_pool and random.random() < p_secondary:
                op2 = random.choice(sec_pool)
                return UnaryOpNode(op2, node)
            return node

        # 根层：基变量 + 一元包裹 + 与常量阈值比较
        if depth == 0:
            base_var = TerminalNode(random.choice(self.dsl_variables))
            # 仅依据允许集合决定是否包一元
            base_var = maybe_unary(base_var, p_primary=0.35, p_secondary=0.10, allow_second_layer=True)
            # 按变量/一元操作类型设置阈值分布（为常见变量采用更保守的阈值范围，避免宽条件）
            if isinstance(base_var, UnaryOpNode):
                opn = base_var.op
                if opn in ('sin', 'cos'):
                    thresh_val = random.uniform(-1.0, 1.0)
                elif opn == 'tan':
                    thresh_val = random.uniform(-5.0, 5.0)
                elif opn == 'log1p':
                    thresh_val = random.uniform(0.0, 2.5)  # log1p(|x|) >= 0
                elif opn == 'sqrt':
                    thresh_val = random.uniform(0.0, 3.0)
                else:  # abs/sign or nested abs of primary
                    base = random.choice(self.dsl_constants)
                    noise = random.uniform(0.5, 1.3)
                    thresh_val = base * noise
            else:
                # 对关键变量采用更保守的随机范围
                if isinstance(base_var, TerminalNode) and isinstance(base_var.value, str):
                    var_name = base_var.value
                    thresh_val = self._sample_threshold_for_variable(var_name)
                else:
                    base = random.choice(self.dsl_constants)
                    noise = random.uniform(0.5, 1.3)
                    thresh_val = base * noise
            thresh = TerminalNode(round(float(thresh_val), 4))
            op = random.choice(comparators)
            root = BinaryOpNode(op, base_var, thresh)
            try:
                return self._enforce_narrow_condition(root)
            except Exception:
                return root

        # 叶：返回变量或常量（优先变量），再做一元包裹
        if depth >= max_depth or random.random() < 0.5:
            if random.random() < 0.8:
                term = TerminalNode(random.choice(self.dsl_variables))
            else:
                term = TerminalNode(random.choice(self.dsl_constants))
            return maybe_unary(term, p_primary=0.40, p_secondary=0.12, allow_second_layer=True)

        # 内部节点：递归生成左右再比较
        left = self._generate_random_condition(depth + 1, max_depth)
        right = self._generate_random_condition(depth + 1, max_depth)
        # 避免左右都是纯常量（使条件失去动态性）
        if isinstance(left, TerminalNode) and isinstance(right, TerminalNode) and not (isinstance(left.value, str) or isinstance(right.value, str)):
            left = TerminalNode(random.choice(self.dsl_variables))
        op = random.choice(comparators)
        return BinaryOpNode(op, left, right)

    # --- Helper: 阈值抽样更保守（避免宽条件）---
    def _sample_threshold_for_variable(self, var_name: str) -> float:
        import random
        # 针对常见变量设置“更保守”的范围；数值可按数据分布微调
        if var_name == 'pos_err_z':
            return random.uniform(0.1, 0.9)   # 原来可到 2~3m，现在收紧到 <1m
        if var_name == 'pos_err_xy':
            return random.uniform(0.2, 1.2)
        if var_name in ('ang_vel_x', 'ang_vel_y', 'ang_vel_mag'):
            return random.uniform(0.3, 2.0)
        if var_name in ('rpy_err_mag', 'pos_err_z_abs'):
            return random.uniform(0.1, 1.5)
        # 积分项容易累积，阈值适度收紧
        if var_name in ('err_i_z','err_i_x','err_i_y'):
            return random.uniform(0.2, 2.0)
        # 默认：略收窄噪声范围
        import math
        base = random.choice(self.dsl_constants)
        return float(base * random.uniform(0.5, 1.2))

    def _estimate_narrowness(self, program: list) -> float:
        """对条件“窄度”打分（0~1）：
        - 对 '<'：阈值越小（>=0）越窄；按每变量 lt_max 归一化：score = 1 - clip(T/lt_max,0,1)
        - 对 '>'：阈值越大（>0）越窄；按 gt_min 做饱和：score = clip(T/(3*gt_min), 0, 1)
        - 统一映射到 [0,1]，遇到负值或无法解析变量则降权。
        """
        if not program:
            return 0.0
        total = 0.0; cnt = 0
        for rule in program:
            cond = rule.get('condition')
            if not (isinstance(cond, BinaryOpNode) and cond.op in ('<','>') and isinstance(cond.right, TerminalNode) and isinstance(cond.right.value,(int,float))):
                continue
            T = float(cond.right.value)
            var_name = self._get_base_var_name(cond.left)
            # trig: 若是 abs(sin/cos(...)) < T，则采用较小的 trig_lt_max 做归一化，鼓励非常窄的相位窗口
            def _has_trig(n: ProgramNode) -> bool:
                cur = n
                while isinstance(cur, UnaryOpNode):
                    if cur.op in ('sin','cos'):
                        return True
                    cur = cur.child
                return False
            # 变量特定阈值参考（与 _enforce_narrow_condition 保持一致）
            strict_caps = {
                'pos_err_z': {'lt_max': 1.0, 'gt_min': 0.15},
                'pos_err_xy': {'lt_max': 1.6, 'gt_min': 0.2},
                'ang_vel_x': {'lt_max': 2.2, 'gt_min': 0.3},
                'ang_vel_y': {'lt_max': 2.2, 'gt_min': 0.3},
                'ang_vel_mag': {'lt_max': 2.5, 'gt_min': 0.4},
                'rpy_err_mag': {'lt_max': 1.8, 'gt_min': 0.2},
                'pos_err_z_abs': {'lt_max': 1.2, 'gt_min': 0.15},
                'err_i_z': {'lt_max': 2.0, 'gt_min': 0.25},
                'err_i_x': {'lt_max': 2.0, 'gt_min': 0.25},
                'err_i_y': {'lt_max': 2.0, 'gt_min': 0.25},
            }
            caps = {'lt_max': 1.5, 'gt_min': 0.2}
            if isinstance(var_name, str) and var_name in strict_caps:
                caps = strict_caps[var_name]
            # trig 情况：若左侧包含 sin/cos 且比较为 '<'，以 trig_lt_max 为 lt_max
            if _has_trig(cond.left) and cond.op == '<':
                caps = {'lt_max': float(getattr(self, '_trig_lt_max', 0.25)), 'gt_min': caps.get('gt_min', 0.2)}
            score = 0.0
            if cond.op == '<':
                T_eff = max(0.0, T)
                ratio = min(1.0, T_eff / max(1e-6, caps['lt_max']))
                score = 1.0 - ratio
            else:  # '>'
                if T <= 0:
                    score = 0.0
                else:
                    # 随 T 增长快速饱和
                    score = min(1.0, T / max(1e-6, 3.0 * caps['gt_min']))
            total += max(0.0, min(1.0, score)); cnt += 1
        return float(total/cnt) if cnt>0 else 0.0

    def _estimate_variable_diversity(self, program: list) -> float:
        # 统计条件中不同变量的覆盖度（简单去重计数 / 归一化）
        if not program:
            return 0.0
        vars_set = set()
        def collect_vars(node):
            if isinstance(node, TerminalNode) and isinstance(node.value, str):
                vars_set.add(node.value)
            elif isinstance(node, UnaryOpNode):
                collect_vars(node.child)
            elif isinstance(node, BinaryOpNode):
                collect_vars(node.left); collect_vars(node.right)
        for rule in program:
            collect_vars(rule.get('condition'))
        # 用“出现变量种类/总可用变量种类”的比例作为分数
        total_vars = max(1, len(self.dsl_variables))
        return min(1.0, len(vars_set) / total_vars)

    def _get_base_var_name(self, node: ProgramNode) -> Optional[str]:
        cur = node
        while isinstance(cur, UnaryOpNode):
            cur = cur.child
        if isinstance(cur, TerminalNode) and isinstance(cur.value, str):
            return cur.value
        return None

    def _enforce_narrow_condition(self, cond: ProgramNode) -> ProgramNode:
        try:
            if isinstance(cond, BinaryOpNode) and cond.op in ('<','>') and isinstance(cond.right, TerminalNode) and isinstance(cond.right.value,(int,float)):
                # 若出现未允许的一元算子，转换为允许形态（优先使用 abs(base_var)）
                def has_disallowed_unary(n: ProgramNode) -> bool:
                    cur = n
                    allowed = getattr(self, '_allowed_cond_unaries', set(['identity','abs']))
                    while isinstance(cur, UnaryOpNode):
                        if cur.op not in allowed:
                            return True
                        cur = cur.child
                    return False
                def has_trig(n: ProgramNode) -> bool:
                    cur = n
                    while isinstance(cur, UnaryOpNode):
                        if cur.op in ('sin','cos'):
                            return True
                        cur = cur.child
                    return False
                var_name = self._get_base_var_name(cond.left)
                T = float(cond.right.value)
                # 若启用“相位窗口”，则对 trig 条件强制规范为 abs(trig(...)) < 小阈值
                if bool(getattr(self, '_trig_as_phase_window', False)) and has_trig(cond.left):
                    cond.left = UnaryOpNode('abs', cond.left if isinstance(cond.left, UnaryOpNode) else cond.left)
                    cond.op = '<'
                    T = min(max(T, 0.02), float(getattr(self, '_trig_lt_max', 0.25)))
                if has_disallowed_unary(cond.left) and var_name is not None:
                    # 替换为 abs(base_var) 与 '<' 比较一个紧的阈值
                    cond.left = UnaryOpNode('abs', TerminalNode(var_name))
                    cond.op = '<'
                if var_name:
                    strict_caps = {
                        'pos_err_z': {'lt_max': 1.0, 'gt_min': 0.15},
                        'pos_err_xy': {'lt_max': 1.6, 'gt_min': 0.2},
                        'ang_vel_x': {'lt_max': 2.2, 'gt_min': 0.3},
                        'ang_vel_y': {'lt_max': 2.2, 'gt_min': 0.3},
                        'ang_vel_mag': {'lt_max': 2.5, 'gt_min': 0.4},
                        'rpy_err_mag': {'lt_max': 1.8, 'gt_min': 0.2},
                        'pos_err_z_abs': {'lt_max': 1.2, 'gt_min': 0.15},
                        'err_i_z': {'lt_max': 2.0, 'gt_min': 0.25},
                        'err_i_x': {'lt_max': 2.0, 'gt_min': 0.25},
                        'err_i_y': {'lt_max': 2.0, 'gt_min': 0.25},
                    }
                    caps = {'lt_max': 1.5, 'gt_min': 0.2}
                    if isinstance(var_name, str) and var_name in strict_caps:
                        caps = strict_caps[var_name]
                    if cond.op == '<':
                        T = min(T, caps['lt_max'])
                        T = max(T, 0.02)
                    else:  # '>'
                        T = max(T, caps['gt_min'])
                    # 若左侧为 abs() 且比较为 '>'，这种条件往往过宽，改为 '<' 并使用较紧阈值
                    if isinstance(cond.left, UnaryOpNode) and cond.left.op == 'abs' and cond.op == '>':
                        cond.op = '<'
                        T = min(T, caps['lt_max'] * 0.6)
                    # trig 相位窗口再收紧一次（若已启用）
                    if bool(getattr(self, '_trig_as_phase_window', False)) and has_trig(cond.left):
                        cond.op = '<'
                        T = min(T, float(getattr(self, '_trig_lt_max', 0.25)))
                    cond.right = TerminalNode(round(T, 4))
        except Exception:
            return cond
        return cond

    def _estimate_strictness(self, program: list) -> float:
        # 严格度沿用改进后的“窄度”打分
        return self._estimate_narrowness(program)

    def _clone_rule(self, rule:dict)->dict:
        # 浅克隆 (AST 节点不可变使用即可)；动作/条件再利用引用足够
        return {'condition': rule['condition'], 'action': list(rule['action'])}
    def _generate_random_action(self)->list:
        """生成数学原语动作，随机设置 1~2 个输出键，每个键的右值为表达式 AST。"""
        import random
        output_keys = ['u_fz','u_tx','u_ty','u_tz']
        k = random.randint(1, 2)
        random.shuffle(output_keys)
        chosen = output_keys[:k]
        acts = []
        for key in chosen:
            expr = self._generate_random_expression()
            acts.append(BinaryOpNode('set', TerminalNode(key), expr))
        return acts

    def _generate_random_expression(self, depth: int = 0, max_depth: int = 3) -> ProgramNode:
        import random
        # 叶子：变量或常数，并可能包一元函数
        if depth >= max_depth or random.random() < 0.35:
            if random.random() < 0.7 and self.dsl_variables:
                node: ProgramNode = TerminalNode(random.choice(self.dsl_variables))
            else:
                node = TerminalNode(random.choice(self.dsl_constants) if self.dsl_constants else 1.0)
            unary_pool = [op for op in self.dsl_operators if op in ('abs','sin','cos','tan','log1p','sqrt')]
            if unary_pool and random.random() < 0.4:
                node = UnaryOpNode(random.choice(unary_pool), node)
            return node
        # 内部：二元组合
        left = self._generate_random_expression(depth+1, max_depth)
        right = self._generate_random_expression(depth+1, max_depth)
        bin_pool = [op for op in self.dsl_operators if op in ('+','-','*','/','max','min')]
        op = random.choice(bin_pool) if bin_pool else '*'
        return BinaryOpNode(op, left, right)
    def _ast_to_str(self,node:ProgramNode)->str:
        if isinstance(node,BinaryOpNode): return f"({self._ast_to_str(node.left)} {node.op} {self._ast_to_str(node.right)})"
        if isinstance(node,UnaryOpNode): return f"{node.op}({self._ast_to_str(node.child)})"
        if isinstance(node,IfNode): return f"IF({self._ast_to_str(node.condition)}) THEN({self._ast_to_str(node.then_branch)}) ELSE({self._ast_to_str(node.else_branch)})"
        if isinstance(node,TerminalNode): return str(node.value)
        return 'UNKNOWN'
    # --- Hashing for transpositions ---
    def _hash_program(self, program: list) -> str:
        parts = []
        for rule in program:
            cond = self._ast_to_str(rule['condition'])
            acts = []
            for a in rule['action']:
                if isinstance(a,BinaryOpNode) and a.op=='set' and isinstance(a.left,TerminalNode):
                    rstr = self._ast_to_str(a.right) if hasattr(a,'right') else '0'
                    acts.append(f"{a.left.value}:{rstr}")
            parts.append(cond+"|"+",".join(sorted(acts)))
        # 将可变 salt 混入哈希，隔离不同评估上下文的 TT 记录（例如不同 duration/批次）
        salt = str(getattr(self, '_tt_salt', ''))
        raw = salt + "::" + "||".join(parts)
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()
