"""在线训练主循环 - AlphaZero式程序合成

从零开始训练：NN随机初始化 → MCTS搜索 → 收集样本 → 更新NN → 循环
"""
from __future__ import annotations

# 【修复Python 3.13兼容性】禁用PyTorch编译功能
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import argparse, time, json, random
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import numpy as np

# 导入现有模块 - 简化导入,只导入必需组件
import sys, pathlib
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Ensure Isaac Gym python bindings are importable (repo vendor path)
try:
    _REPO_ROOT = _PKG_ROOT.parent
    _GYM_PY = _REPO_ROOT / 'isaacgym' / 'python'
    if _GYM_PY.exists() and str(_GYM_PY) not in sys.path:
        sys.path.insert(0, str(_GYM_PY))
    # 提前导入 isaacgym，确保其先于 torch 导入
    try:
        from isaacgym import gymapi  # type: ignore
    except Exception:
        pass
except Exception:
    pass

# 直接导入必需模块（避免循环依赖）
from mcts_training.mcts import MCTS_Agent, MCTSNode
from mcts_training.policy.policy_nn import PolicyValueNNLarge, EDIT_TYPES
from mcts_training.program_features import featurize_program

# GNN相关模块（可选）
try:
    from gnn_features import ast_to_pyg_graph, batch_programs_to_graphs
    from gnn_policy_nn import GNNPolicyValueNet as GNNPolicyValueNetV1
    # v2 可选导入
    try:
        from gnn_policy_nn_v2 import create_gnn_policy_value_net_v2 as create_gnn_policy_value_net_v2
        GNN_V2_AVAILABLE = True
    except ImportError:
        create_gnn_policy_value_net_v2 = None  # type: ignore
        GNN_V2_AVAILABLE = False
    from torch_geometric.data import Batch as PyGBatch
    GNN_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] GNN模块不可用: {e}")
    GNN_AVAILABLE = False
    GNN_V2_AVAILABLE = False
    ast_to_pyg_graph = None
    batch_programs_to_graphs = None
    GNNPolicyValueNetV1 = None  # type: ignore
    create_gnn_policy_value_net_v2 = None  # type: ignore
    PyGBatch = None

# 导入batch_evaluation（可能需要Isaac Gym）；确保在导入 torch 之前尝试导入 isaacgym
try:
    from batch_evaluation import BatchEvaluator
    BATCH_EVAL_AVAILABLE = True
except Exception as e:
    print(f"[Warning] BatchEvaluator不可用: {e}")
    BATCH_EVAL_AVAILABLE = False
    BatchEvaluator = None  # type: ignore

# 现在再导入 torch 及其子模块，避免破坏 isaacgym 的导入顺序要求
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 无Isaac Gym时的简易占位Evaluator（仅用于开发/单元测试，不代表真实性能）
class _DummyEvaluator:
    def __init__(self, *args, **kwargs) -> None:
        self._rng = random.Random(0)
    def evaluate_single(self, program: List[Dict[str, Any]]) -> float:
        # 粗略按规则数给一点偏好，仍保留随机性，便于跑通流程
        base = float(len(program)) * 0.05
        return base + (self._rng.random() - 0.5) * 0.1
    def evaluate_batch(self, programs: List[List[Dict[str, Any]]]):
        return [self.evaluate_single(p) for p in programs]

# 导入serialization
try:
    from serialization import save_program_json as _save_prog
    def save_program_json(program, path):  # type: ignore
        _save_prog(program, path)
except Exception:
    def save_program_json(program, path):  # type: ignore
        import json
        # 简化版保存（不包含节点对象）
        simplified = []
        for rule in program:
            simple_rule = {
                'name': rule.get('name', 'rule'),
                'multiplier': rule.get('multiplier', [1.0, 1.0, 1.0])
            }
            simplified.append(simple_rule)
        
        with open(path, 'w') as f:
            json.dump({'rules': simplified, 'note': 'Simplified format'}, f, indent=2)


class ReplayBuffer:
    """经验回放缓冲区（支持固定特征和GNN图数据）"""
    
    def __init__(self, capacity: int = 50000, use_gnn: bool = False):
        self.capacity = capacity
        self.use_gnn = use_gnn
        self.buffer = deque(maxlen=capacity)
    
    def push(self, sample: Dict[str, Any]):
        """添加样本
        
        固定特征模式: sample = {'features': tensor, 'policy_target': tensor, 'value_target': tensor}
        GNN模式: sample = {'graph': PyG Data, 'policy_target': tensor, 'value_target': tensor}
        """
        self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """随机采样"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class OnlineTrainer:
    """在线训练器 - AlphaZero范式"""
    
    def __init__(self, args):
        self.args = args
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] 使用设备: {self.device}")
        
        # 检查是否使用GNN
        self.use_gnn = getattr(args, 'use_gnn', False) and GNN_AVAILABLE
        if args.use_gnn and not GNN_AVAILABLE:
            print("[Warning] --use-gnn 指定但GNN模块不可用，回退到固定特征网络")
            self.use_gnn = False
        
        # 初始化NN（根据use_gnn选择模型）
        if self.use_gnn:
            # 选择版本
            nn_version = getattr(args, 'nn_version', 'v1')
            if nn_version == 'v2' and GNN_V2_AVAILABLE:
                print(f"[Trainer] 使用 GNN v2 (Hierarchical Dual) 网络")
                self.nn_model = create_gnn_policy_value_net_v2(
                    node_feature_dim=24,
                    policy_output_dim=len(EDIT_TYPES),
                    structure_hidden=256,
                    structure_layers=5,
                    structure_heads=8,
                    feature_layers=3,
                    feature_heads=8,
                    dropout=0.1
                ).to(self.device)
            else:
                if nn_version == 'v2' and not GNN_V2_AVAILABLE:
                    print("[Trainer] 请求 v2 但未找到模块，回退到 v1")
                else:
                    print(f"[Trainer] 使用 GNN v1 网络")
                self.nn_model = GNNPolicyValueNetV1(
                    node_feature_dim=24,
                    hidden_dim=args.nn_hidden,
                    num_layers=3,
                    num_heads=4,
                    policy_output_dim=len(EDIT_TYPES),
                    dropout=0.1
                ).to(self.device)
        else:
            print(f"[Trainer] 使用固定特征策略-价值网络")
            self.nn_model = PolicyValueNNLarge(
                in_dim=64,
                hidden=args.nn_hidden,
                out_dim=len(EDIT_TYPES)
            ).to(self.device)
        
        # 禁用torch compile避免Python 3.13兼容性问题
        try:
            import os
            os.environ['PYTORCH_JIT'] = '0'
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
        except Exception:
            pass
        
        try:
            self.optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=args.learning_rate,
                weight_decay=1e-4
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # 如果标准Adam失败，尝试手动创建
            print(f"[Warning] Adam初始化失败，使用简化版: {e}")
            self.optimizer = optim.SGD(
                self.nn_model.parameters(),
                lr=args.learning_rate,
                momentum=0.9
            )
        
        print(f"[Trainer] NN初始化完成 (参数: {sum(p.numel() for p in self.nn_model.parameters())})")
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=args.replay_capacity, use_gnn=self.use_gnn)
        
        # 评估器：支持强制使用 Dummy，用于快速A/B基准
        force_dummy = getattr(args, 'use_dummy_eval', False)
        if force_dummy or BatchEvaluator is None:
            if not force_dummy:
                print("[Trainer] 使用 DummyEvaluator（未检测到 Isaac Gym）")
            else:
                print("[Trainer] 强制使用 DummyEvaluator（A/B快速基准）")
            self.evaluator = _DummyEvaluator()
        else:
            self.evaluator = BatchEvaluator(
                trajectory_config=self._build_trajectory(),
                duration=args.duration,
                isaac_num_envs=args.isaac_num_envs,
                device=str(self.device),
                replicas_per_program=getattr(args, 'eval_replicas_per_program', 1),
                min_steps_frac=getattr(args, 'min_steps_frac', 0.0),
                reward_reduction=getattr(args, 'reward_reduction', 'sum'),
                strict_no_prior=False,  # ✅ 允许使用状态变量进行反馈控制!
                zero_action_penalty=1.5,
                use_fast_path=getattr(args, 'use_fast_path', False)
            )
        
        # 统计
        self.iteration = 0
        self.best_reward = -float('inf')
        self.best_program = None
        self.training_stats = []
        self._mcts_stats = {}  # MCTS性能统计
    
    def _build_trajectory(self) -> Dict[str, Any]:
        """构建轨迹配置"""
        if self.args.traj == 'hover':
            return {'type': 'hover', 'initial_xyz': [0, 0, 1.0], 'params': {}}
        elif self.args.traj == 'figure8':
            return {'type': 'figure8', 'initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8, 'B': 0.5, 'period': 12}}
        elif self.args.traj == 'circle':
            return {'type': 'circle', 'initial_xyz': [0, 0, 0.8], 'params': {'R': 0.9, 'period': 10}}
        elif self.args.traj == 'helix':
            return {'type': 'helix', 'initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7, 'period': 10, 'v_z': 0.15}}
        else:
            raise ValueError(f"Unknown trajectory: {self.args.traj}")
    
    def _generate_random_program(self) -> List[Dict[str, Any]]:
        """生成随机初始程序"""
        # 使用MCTS的随机生成逻辑
        mcts = MCTS_Agent(
            evaluation_function=lambda p: 0.0,  # 占位符
            dsl_variables=['pos_err', 'vel_err'],
            dsl_constants=[0.0, 1.0],
            dsl_operators=['+', '-', '*']
        )
        return mcts._generate_random_segmented_program()
    
    def _load_program_from_json(self, path: str) -> Optional[List[Dict[str, Any]]]:
        """从 JSON 文件加载程序（用于 warm start）"""
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            
            # 尝试提取 rules 字段
            if isinstance(data, dict) and 'rules' in data:
                rules = data['rules']
            elif isinstance(data, list):
                rules = data
            else:
                print(f"[Warning] 无法解析程序文件格式: {path}")
                return None
            
            # 简单验证
            if not isinstance(rules, list) or len(rules) == 0:
                print(f"[Warning] 程序文件为空或格式错误: {path}")
                return None
            
            print(f"[Trainer] ✅ 从 {path} 加载了 {len(rules)} 条规则")
            return rules
            
        except FileNotFoundError:
            print(f"[Warning] 程序文件不存在: {path}")
            return None
        except Exception as e:
            print(f"[Warning] 加载程序文件失败: {e}")
            return None
    
    def mcts_search(self, root_program: List[Dict[str, Any]], num_simulations: int = 800) -> Tuple[List[Any], List[int]]:
        """
        执行MCTS搜索（使用当前NN引导）
        
        Returns:
            children: 所有子节点
            visit_counts: 访问次数分布
        """
        # 创建MCTS agent
        mcts = MCTS_Agent(
            evaluation_function=self.evaluator.evaluate_single,
            # 使用底层状态变量，提升表达力（严格零先验，不引入 PID 增益语义）
            dsl_variables=[
                'pos_err_x','pos_err_y','pos_err_z','pos_err_xy','pos_err_z_abs',
                'vel_x','vel_y','vel_z','vel_err',
                'ang_vel_x','ang_vel_y','ang_vel_z','ang_vel','ang_vel_mag',
                'err_i_x','err_i_y','err_i_z',
                'err_p_roll','err_p_pitch','err_p_yaw','rpy_err_mag',
                'err_d_x','err_d_y','err_d_z','err_d_roll','err_d_pitch','err_d_yaw'
            ],
            # 常数基数更细，利于数值缩放
            dsl_constants=[0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
            # 表达式算子：保留基础代数 + 简单幅度压缩，不在条件中引入三角（条件生成器已有安全约束）
            dsl_operators=['+','-','*','/','max','min','abs','sqrt','log1p','>','<'],
            exploration_weight=self.args.exploration_weight,
            max_depth=self.args.max_depth
        )
        
        # 设置root
        root = MCTSNode(root_program, parent=None, depth=0)
        mcts.root = root
        
        # 🔧 优化1: GNN先验缓存 (避免重复推理)
        gnn_prior_cache = {}  # prog_hash -> (prior_p, value_estimate)
        
        def get_program_hash(program):
            """生成程序的哈希值用于缓存（使用程序长度+字符串表示）"""
            try:
                # 简单但有效的哈希: 程序长度 + 规则数 + 字符串表示的哈希
                prog_str = str(program)
                return hash((len(program), prog_str))
            except:
                # 回退：使用id（不缓存）
                return id(program)
        
        # 🔧 优化2: 批量GNN推理缓冲区
        pending_gnn_nodes = []  # 收集需要GNN推理的新节点
        
        # 🔧 批量评估优化：收集待评估的leaf nodes
        pending_evals = []  # [(leaf, path, use_real_sim)]
        
        # 执行MCTS模拟（只做树扩展，延迟GNN推理）
        for sim_idx in range(num_simulations):
            # Selection + Expansion（使用NN先验）
            node = root
            path = [node]
            
            # Selection阶段
            while node.children and not node.is_fully_expanded():
                # 使用PUCT选择（集成NN先验）
                node = self._select_child_puct(node)
                path.append(node)
            
            # Expansion阶段
            if not node.is_fully_expanded():
                # 生成新子节点，分配NN先验
                mcts._ensure_mutations(node)
                
                if node.untried_mutations and len(node.expanded_actions) < len(node.untried_mutations):
                    # 选择一个未扩展的变异
                    unexpanded_idx = [i for i in range(len(node.untried_mutations)) 
                                     if i not in node.expanded_actions][0]
                    mutation = node.untried_mutations[unexpanded_idx]
                    
                    # 克隆程序并应用变异
                    child_program = [mcts._clone_rule(r) for r in node.program]
                    mcts._apply_mutation(child_program, mutation)
                    
                    # 创建子节点
                    child = MCTSNode(child_program, parent=node, depth=node.depth + 1)
                    edit_type = mutation[0]
                    child._edit_type = edit_type
                    
                    # 🚀 优化: 检查缓存
                    prog_hash = get_program_hash(child_program)
                    if prog_hash in gnn_prior_cache:
                        # 命中缓存，直接使用
                        child._prior_p, child._cached_value = gnn_prior_cache[prog_hash]
                    else:
                        # 未命中，加入批量推理队列
                        child._prior_p = 1.0 / len(EDIT_TYPES)  # 默认先验
                        child._cached_value = None
                        child._prog_hash = prog_hash
                        pending_gnn_nodes.append((child, edit_type))
                    
                    node.children.append(child)
                    node.expanded_actions.add(unexpanded_idx)
                    path.append(child)
            
            # 🔧 收集leaf待批量评估（不立即评估）
            leaf = path[-1]
            use_real_sim = random.random() < getattr(self.args, 'real_sim_frac', 0.8)
            pending_evals.append((leaf, path, use_real_sim))
        
        # 🚀 批量GNN推理阶段 (一次推理所有新节点)
        if pending_gnn_nodes:
            try:
                with torch.no_grad():
                    if self.use_gnn:
                        # 批量构建图
                        graphs = [ast_to_pyg_graph(child.program) for child, _ in pending_gnn_nodes]
                        from torch_geometric.data import Batch
                        batch_graph = Batch.from_data_list(graphs).to(self.device)
                        policy_logits, value_preds = self.nn_model(batch_graph)
                    else:
                        # 批量特征化
                        features = torch.stack([featurize_program(child.program) 
                                               for child, _ in pending_gnn_nodes]).to(self.device)
                        policy_logits, value_preds = self.nn_model(features)
                    
                    # 分配先验和缓存
                    policy_probs = F.softmax(policy_logits, dim=-1)
                    for idx, (child, edit_type) in enumerate(pending_gnn_nodes):
                        if edit_type in EDIT_TYPES:
                            type_idx = EDIT_TYPES.index(edit_type)
                            prior_p = policy_probs[idx, type_idx].item()
                        else:
                            prior_p = 1.0 / len(EDIT_TYPES)
                        
                        value_est = value_preds[idx].item() if value_preds.dim() > 0 else value_preds.item()
                        child._prior_p = prior_p
                        child._cached_value = value_est
                        
                        # 更新缓存
                        if hasattr(child, '_prog_hash'):
                            gnn_prior_cache[child._prog_hash] = (prior_p, value_est)
            except Exception as e:
                # 批量推理失败，使用默认值
                for child, _ in pending_gnn_nodes:
                    child._prior_p = 1.0 / len(EDIT_TYPES)
                    child._cached_value = None
        
        # 🔧 批量评估阶段
        # 分离真实仿真和NN估值
        real_sim_leaves = [(leaf, path) for leaf, path, use_real in pending_evals if use_real]
        nn_sim_leaves = [(leaf, path) for leaf, path, use_real in pending_evals if not use_real]
        
        # 批量真实仿真
        if real_sim_leaves:
            programs = [leaf.program for leaf, _ in real_sim_leaves]
            rewards = self.evaluator.evaluate_batch(programs)
            for (leaf, path), reward in zip(real_sim_leaves, rewards):
                for node in reversed(path):
                    node.visits += 1
                    node.value_sum += reward
        
        # 🚀 批量NN估值 (使用缓存 + 批量推理)
        if nn_sim_leaves:
            # 检查哪些已有缓存值
            cached_leaves = []
            uncached_leaves = []
            for leaf, path in nn_sim_leaves:
                if hasattr(leaf, '_cached_value') and leaf._cached_value is not None:
                    # 使用缓存的value
                    cached_leaves.append((leaf, path, leaf._cached_value * 10.0))
                else:
                    # 需要批量推理
                    prog_hash = get_program_hash(leaf.program)
                    if prog_hash in gnn_prior_cache:
                        _, value_est = gnn_prior_cache[prog_hash]
                        cached_leaves.append((leaf, path, value_est * 10.0))
                    else:
                        uncached_leaves.append((leaf, path))
            
            # 处理缓存命中的
            for leaf, path, reward in cached_leaves:
                for node in reversed(path):
                    node.visits += 1
                    node.value_sum += reward
            
            # 批量推理未缓存的
            if uncached_leaves:
                try:
                    with torch.no_grad():
                        if self.use_gnn:
                            graphs = [ast_to_pyg_graph(leaf.program) for leaf, _ in uncached_leaves]
                            from torch_geometric.data import Batch
                            batch_graph = Batch.from_data_list(graphs).to(self.device)
                            _, value_preds = self.nn_model(batch_graph)
                        else:
                            features = torch.stack([featurize_program(leaf.program) 
                                                   for leaf, _ in uncached_leaves]).to(self.device)
                            _, value_preds = self.nn_model(features)
                        
                        # 分配rewards并更新缓存
                        for idx, (leaf, path) in enumerate(uncached_leaves):
                            value_est = value_preds[idx].item() if value_preds.dim() > 0 else value_preds.item()
                            reward = value_est * 10.0
                            
                            # 更新缓存
                            prog_hash = get_program_hash(leaf.program)
                            gnn_prior_cache[prog_hash] = (1.0 / len(EDIT_TYPES), value_est)
                            
                            for node in reversed(path):
                                node.visits += 1
                                node.value_sum += reward
                except Exception:
                    # 批量推理失败，使用默认值
                    for leaf, path in uncached_leaves:
                        reward = -10.0
                        for node in reversed(path):
                            node.visits += 1
                            node.value_sum += reward
        
        # 📊 性能统计 (可选，用于调试)
        if hasattr(self, '_mcts_stats'):
            total_gnn_calls = len(pending_gnn_nodes) + len(uncached_leaves if 'uncached_leaves' in locals() else [])
            cached_hits = len(cached_leaves if 'cached_leaves' in locals() else [])
            self._mcts_stats['total_gnn_nodes'] = self._mcts_stats.get('total_gnn_nodes', 0) + len(pending_gnn_nodes)
            self._mcts_stats['total_value_cached'] = self._mcts_stats.get('total_value_cached', 0) + cached_hits
            self._mcts_stats['cache_size'] = len(gnn_prior_cache)
        
        # 返回root的子节点和访问分布
        if root.children:
            visit_counts = [child.visits for child in root.children]
            return root.children, visit_counts
        else:
            return [], []
    
    def _select_child_puct(self, node: MCTSNode) -> MCTSNode:
        """PUCT选择（使用NN先验）"""
        if not node.children:
            return node
        
        best_score = -float('inf')
        best_child = None
        
        sqrt_n = np.sqrt(node.visits)
        c_puct = self.args.puct_c
        
        for child in node.children:
            # Q值：平均奖励
            q = child.value_sum / child.visits if child.visits > 0 else 0.0
            
            # U值：探索奖励（使用NN先验）
            prior = getattr(child, '_prior_p', 1.0 / len(node.children))
            u = c_puct * prior * sqrt_n / (1 + child.visits)
            
            score = q + u
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child else node.children[0]
    
    def train_step(self):
        """单步训练"""
        if len(self.replay_buffer) < self.args.batch_size:
            return
        
        # 采样batch
        batch = self.replay_buffer.sample(self.args.batch_size)
        
        # 构建tensor（根据模式）
        if self.use_gnn:
            # GNN模式：使用PyG Batch
            graph_list = [s['graph'] for s in batch]
            batch_graph = PyGBatch.from_data_list(graph_list).to(self.device)
            policy_targets = torch.stack([s['policy_target'] for s in batch]).to(self.device)
            value_targets = torch.stack([s['value_target'] for s in batch]).to(self.device)
            
            # 前向传播
            policy_logits, value_preds = self.nn_model(batch_graph)
        else:
            # 固定特征模式
            features = torch.stack([s['features'] for s in batch]).to(self.device)
            policy_targets = torch.stack([s['policy_target'] for s in batch]).to(self.device)
            value_targets = torch.stack([s['value_target'] for s in batch]).to(self.device)
            
            # 前向传播
            policy_logits, value_preds = self.nn_model(features)
        
        # 损失计算
        # 策略损失：交叉熵（MCTS访问分布作为目标）
        policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        
        # 价值损失：MSE
        value_loss = F.mse_loss(value_preds.squeeze(), value_targets.squeeze())
        
        # 总损失
        total_loss = policy_loss + self.args.value_loss_weight * value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self):
        """主训练循环"""
        print(f"\n{'='*80}")
        print(f"开始在线训练 - AlphaZero式程序合成")
        print(f"{'='*80}")
        print(f"总迭代数: {self.args.total_iters}")
        print(f"MCTS模拟数/迭代: {self.args.mcts_simulations}")
        print(f"NN更新频率: 每{self.args.update_freq}次迭代")
        print(f"批量大小: {self.args.batch_size}")
        print(f"{'='*80}\n")
        
        # 初始化程序（支持从文件加载）
        if hasattr(self.args, 'warm_start') and self.args.warm_start:
            loaded_program = self._load_program_from_json(self.args.warm_start)
            if loaded_program:
                current_program = loaded_program
                print(f"[Trainer] 🔥 Warm Start: 使用预训练程序 ({len(current_program)} 条规则)")
            else:
                current_program = self._generate_random_program()
                print(f"[Trainer] ⚠️ Warm Start 失败，使用随机初始化")
        else:
            current_program = self._generate_random_program()
        
        for iter_idx in range(self.args.total_iters):
            iter_start_time = time.time()
            
            print(f"\n[Iter {iter_idx+1}/{self.args.total_iters}] MCTS搜索中...")
            
            # MCTS搜索
            children, visit_counts = self.mcts_search(current_program, self.args.mcts_simulations)
            
            if not children:
                print(f"[Iter {iter_idx+1}] ⚠️ 未生成子节点，跳过")
                continue
            
            # 选择访问最多的子节点
            best_child_idx = np.argmax(visit_counts)
            best_child = children[best_child_idx]
            next_program = best_child.program
            
            # 真实评估（每次迭代至少1次）
            reward = self.evaluator.evaluate_single(next_program)
            
            # 收集训练样本
            # 策略标签：将根子节点访问分布按其编辑类型聚合到 EDIT_TYPES
            total_visits = sum(visit_counts)
            policy_target = torch.zeros(len(EDIT_TYPES))
            if total_visits > 0:
                for i, child in enumerate(children):
                    prob = float(visit_counts[i]) / float(total_visits)
                    et = getattr(child, '_edit_type', None)
                    if et in EDIT_TYPES:
                        policy_target[EDIT_TYPES.index(et)] += prob
                    else:
                        # 若未知类型，等量分摊到所有维度，避免丢失概率质量
                        policy_target += prob / len(EDIT_TYPES)
                # 归一化（数值安全）
                s = float(policy_target.sum().item())
                if s > 0:
                    policy_target = policy_target / s
            else:
                # 没有访问计数时，退化为均匀分布
                policy_target += 1.0 / len(EDIT_TYPES)
            
            # 价值标签：归一化奖励
            value_target = torch.tensor([reward / 10.0], dtype=torch.float32)  # 缩放到 [-1, 1]
            
            # 构建样本（根据模式选择特征或图）
            if self.use_gnn:
                sample = {
                    'graph': ast_to_pyg_graph(current_program),
                    'policy_target': policy_target,
                    'value_target': value_target
                }
            else:
                sample = {
                    'features': featurize_program(current_program),
                    'policy_target': policy_target,
                    'value_target': value_target
                }
            
            self.replay_buffer.push(sample)
            
            # 更新NN（每N次迭代）
            if (iter_idx + 1) % self.args.update_freq == 0:
                print(f"[Iter {iter_idx+1}] 更新NN...")
                for _ in range(self.args.train_steps_per_update):
                    losses = self.train_step()
                    if losses:
                        print(f"  Loss: policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}")
            
            # 更新最佳程序
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_program = next_program
                print(f"[Iter {iter_idx+1}] 🎉 新最佳！奖励: {reward:.4f}")
                
                # 保存
                save_program_json(self.best_program, self.args.save_path)
            
            # 更新当前程序
            current_program = next_program
            
            iter_time = time.time() - iter_start_time
            
            # 📊 MCTS性能统计 (每10轮输出一次)
            mcts_info = ""
            if self._mcts_stats and (iter_idx + 1) % 10 == 0:
                total_gnn = self._mcts_stats.get('total_gnn_nodes', 0)
                total_cached = self._mcts_stats.get('total_value_cached', 0)
                cache_size = self._mcts_stats.get('cache_size', 0)
                if total_gnn > 0:
                    hit_rate = total_cached / (total_gnn + total_cached) * 100 if (total_gnn + total_cached) > 0 else 0
                    mcts_info = f" | GNN: {total_gnn}节点 | 缓存命中: {hit_rate:.0f}% ({cache_size}项)"
                # 重置统计
                self._mcts_stats = {}
            
            print(f"[Iter {iter_idx+1}] 完成 | 奖励: {reward:.4f} | 耗时: {iter_time:.1f}s | Buffer: {len(self.replay_buffer)}{mcts_info}")
            
            # 定期保存检查点
            if (iter_idx + 1) % self.args.checkpoint_freq == 0:
                checkpoint_path = f"{self.args.save_path.replace('.json', '')}_nn_iter_{iter_idx+1}.pt"
                torch.save(self.nn_model.state_dict(), checkpoint_path)
                print(f"[Iter {iter_idx+1}] 💾 检查点已保存: {checkpoint_path}")
        
        print(f"\n{'='*80}")
        print(f"训练完成！最佳奖励: {self.best_reward:.4f}")
        print(f"{'='*80}\n")


def parse_args():
    p = argparse.ArgumentParser(description='在线训练 - AlphaZero式程序合成')
    
    # 训练参数
    p.add_argument('--total-iters', type=int, default=5000, help='总迭代数')
    p.add_argument('--mcts-simulations', type=int, default=800, help='每次迭代的MCTS模拟数')
    p.add_argument('--update-freq', type=int, default=50, help='NN更新频率')
    p.add_argument('--train-steps-per-update', type=int, default=10, help='每次更新的训练步数')
    p.add_argument('--batch-size', type=int, default=256, help='批量大小')
    p.add_argument('--replay-capacity', type=int, default=50000, help='经验回放容量')
    
    # NN参数
    p.add_argument('--use-gnn', action='store_true', help='使用GNN网络（GAT）代替固定特征网络')
    p.add_argument('--nn-version', type=str, default='v1', choices=['v1','v2'], help='GNN版本: v1(原始) 或 v2(分层双网络)')
    p.add_argument('--nn-hidden', type=int, default=256, help='NN隐藏层维度')
    p.add_argument('--learning-rate', type=float, default=1e-3, help='学习率')
    p.add_argument('--value-loss-weight', type=float, default=0.5, help='价值损失权重')
    
    # MCTS参数
    p.add_argument('--exploration-weight', type=float, default=1.4, help='UCB探索权重')
    p.add_argument('--puct-c', type=float, default=1.5, help='PUCT常数')
    p.add_argument('--max-depth', type=int, default=20, help='MCTS最大深度')
    p.add_argument('--real-sim-frac', type=float, default=0.8, help='MCTS模拟中使用真实仿真的比例 [0,1]，默认0.8保证数据质量')
    
    # 仿真参数（仅Isaac Gym）
    p.add_argument('--traj', type=str, default='figure8', choices=['hover', 'figure8', 'circle', 'helix'])
    p.add_argument('--duration', type=int, default=10, help='仿真时长（秒）')
    p.add_argument('--isaac-num-envs', type=int, default=512, help='Isaac Gym并行环境数')
    p.add_argument('--eval-replicas-per-program', type=int, default=1, help='evaluate_single 时并行副本数，取平均以提高利用率/稳定性')
    p.add_argument('--min-steps-frac', type=float, default=0.0, help='每次评估至少执行的步数比例 [0,1]，避免过早 done 退出')
    p.add_argument('--reward-reduction', type=str, default='sum', choices=['sum','mean'], help="奖励归约方式：'sum'（步次求和）或 'mean'（步次平均）")
    p.add_argument('--use-fast-path', action='store_true', help='启用超高性能优化路径（环境池复用+Numba JIT编译，7×加速）')
    p.add_argument('--use-dummy-eval', action='store_true', help='强制使用Dummy评估器（禁用Isaac Gym），用于快速A/B基准')
    
    # 保存参数
    p.add_argument('--save-path', type=str, default='01_pi_flight/results/online_best_program.json')
    p.add_argument('--checkpoint-freq', type=int, default=50, help='检查点保存频率（默认50）')
    p.add_argument('--warm-start', type=str, default=None, help='从已有程序文件开始训练（JSON 路径）')
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 开始训练
    trainer = OnlineTrainer(args)
    trainer.train()
