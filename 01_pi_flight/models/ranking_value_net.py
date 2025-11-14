"""Ranking-based Value Network for Program Synthesis

基于论文："Ranking Policy Gradient" (Kwon et al., 2019)
核心思想：使用程序的相对排序而非绝对奖励来训练价值网络

解决的问题：
1. 奖励恒定（-59.48）导致无法区分程序优劣
2. Policy loss为0（所有程序被认为同样好）
3. MCTS搜索退化（visit_counts全0或均匀）

方法：
- 收集程序对：(prog_better, prog_worse)
- 训练网络预测：P(prog_a > prog_b)
- 使用学习到的排序分数引导MCTS
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from collections import deque
import random


class RankingValueNet(nn.Module):
    """基于排序的价值网络（整合动作特征）
    
    与标准value网络的区别：
    - 输入：两个程序的GNN嵌入 + 动作幅度特征
    - 输出：prog_a比prog_b更好的概率
    - 训练：使用pairwise ranking loss
    
    动作特征整合：
    - 显式输入动作统计量（mean/std/max的fz和tx）
    - 网络自动学习"零动作程序排名低"
    - 避免手工设计惩罚项
    """
    
    def __init__(self, embed_dim: int = 128, action_feature_dim: int = 6):
        """
        Args:
            embed_dim: GNN嵌入维度
            action_feature_dim: 动作特征维度（默认6：fz_mean/std/max, tx_mean/std/max）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.action_feature_dim = action_feature_dim
        
        # 比较网络：融合两个程序的特征（嵌入+动作）
        # 输入：[embed_a, action_a, embed_b, action_b]
        input_dim = (embed_dim + action_feature_dim) * 2
        self.compare_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出logit（prog_a更好的log-odds）
        )
        
        # 单独的价值头：用于MCTS（嵌入+动作→绝对分数）
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim + action_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出[-1, 1]
        )
    
    def forward_compare(self, embed_a: torch.Tensor, embed_b: torch.Tensor,
                       action_feat_a: Optional[torch.Tensor] = None,
                       action_feat_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """比较两个程序（整合动作特征）
        
        Args:
            embed_a: 程序A的GNN嵌入 [B, embed_dim]
            embed_b: 程序B的GNN嵌入 [B, embed_dim]
            action_feat_a: 程序A的动作特征 [B, action_feature_dim]，可选
                          (fz_mean, fz_std, fz_max, tx_mean, tx_std, tx_max)
            action_feat_b: 程序B的动作特征 [B, action_feature_dim]，可选
        
        Returns:
            logits: 程序A比B更好的log-odds [B, 1]
        
        Note:
            网络通过动作特征自动学习：
            - 大推力程序 > 零推力程序
            - 稳定控制(低std) > 抖动控制(高std)
            - 无需手工设计零动作惩罚
            
            如果action_feat未提供，使用零特征（退化模式）
        """
        # 处理可选动作特征
        if action_feat_a is None:
            action_feat_a = torch.zeros(embed_a.size(0), self.action_feature_dim, 
                                       device=embed_a.device, dtype=embed_a.dtype)
        if action_feat_b is None:
            action_feat_b = torch.zeros(embed_b.size(0), self.action_feature_dim,
                                       device=embed_b.device, dtype=embed_b.dtype)
        
        input_a = torch.cat([embed_a, action_feat_a], dim=-1)  # [B, embed_dim+6]
        input_b = torch.cat([embed_b, action_feat_b], dim=-1)  # [B, embed_dim+6]
        combined = torch.cat([input_a, input_b], dim=-1)      # [B, 2*(embed_dim+6)]
        return self.compare_net(combined)
    
    def forward_value(self, embed: torch.Tensor, action_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """单个程序的绝对价值估计（用于MCTS，整合动作特征）
        
        Args:
            embed: 程序的GNN嵌入 [B, embed_dim]
            action_feat: 动作特征 [B, action_feature_dim]，可选
        
        Returns:
            value: 价值估计 [B, 1]，范围[-1, 1]
        """
        if action_feat is None:
            action_feat = torch.zeros(embed.size(0), self.action_feature_dim,
                                     device=embed.device, dtype=embed.dtype)
        combined = torch.cat([embed, action_feat], dim=-1)
        return self.value_head(combined)
    
    def forward(self, embed: torch.Tensor, action_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """默认forward：返回绝对价值（兼容现有代码）"""
        return self.forward_value(embed, action_feat)


class PairwiseRankingBuffer:
    """存储程序对的replay buffer（整合动作特征）
    
    结构：(prog_a, prog_b, preference, action_feat_a, action_feat_b)
    - prog_a, prog_b: 程序的graph数据
    - preference: 1 if reward_a > reward_b, else 0
    - action_feat_a/b: 动作幅度特征 [fz_mean, fz_std, fz_max, tx_mean, tx_std, tx_max]
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, prog_a_graph, prog_b_graph, preference: float,
             action_feat_a: List[float], action_feat_b: List[float]):
        """添加一对程序（包含动作特征）
        
        Args:
            prog_a_graph: PyG Data对象
            prog_b_graph: PyG Data对象
            preference: 1.0 if a>b, 0.5 if a=b, 0.0 if a<b
            action_feat_a: 程序A的动作特征 [6]
            action_feat_b: 程序B的动作特征 [6]
        """
        self.buffer.append({
            'prog_a': prog_a_graph,
            'prog_b': prog_b_graph,
            'preference': preference,
            'action_feat_a': action_feat_a,
            'action_feat_b': action_feat_b
        })
    
    def sample(self, batch_size: int) -> List[dict]:
        """采样一批程序对"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


def compute_ranking_loss(
    ranking_net: RankingValueNet,
    batch: List[dict],
    gnn_encoder,
    device: torch.device
) -> Tuple[torch.Tensor, dict]:
    """计算ranking loss（整合动作特征）
    
    Args:
        ranking_net: RankingValueNet实例
        batch: 从PairwiseRankingBuffer采样的batch
               每个item包含: prog_a, prog_b, preference, action_feat_a, action_feat_b
        gnn_encoder: GNN编码器（提取程序嵌入）
        device: torch device
    
    Returns:
        loss: ranking loss
        metrics: 训练指标字典
    """
    from torch_geometric.data import Batch as PyGBatch
    
    # 构建批次
    graphs_a = [item['prog_a'] for item in batch]
    graphs_b = [item['prog_b'] for item in batch]
    preferences = torch.tensor(
        [item['preference'] for item in batch],
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)
    
    # 提取动作特征
    action_feats_a = torch.tensor(
        [item['action_feat_a'] for item in batch],
        dtype=torch.float32,
        device=device
    )  # [B, 6]
    action_feats_b = torch.tensor(
        [item['action_feat_b'] for item in batch],
        dtype=torch.float32,
        device=device
    )  # [B, 6]
    
    # 批量GNN编码
    batch_a = PyGBatch.from_data_list(graphs_a).to(device)
    batch_b = PyGBatch.from_data_list(graphs_b).to(device)
    
    # 提取嵌入
    with torch.no_grad():
        embed_a = gnn_encoder.get_embedding(batch_a)
        embed_b = gnn_encoder.get_embedding(batch_b)
    
    # 比较（传入动作特征）
    logits = ranking_net.forward_compare(embed_a, embed_b, action_feats_a, action_feats_b)
    probs = torch.sigmoid(logits)
    
    # Pairwise ranking loss (binary cross entropy)
    loss = F.binary_cross_entropy_with_logits(logits, preferences)
    
    # 计算准确率（预测是否与真实偏好一致）
    predictions = (probs > 0.5).float()
    accuracy = (predictions == preferences).float().mean()
    
    # 分析动作特征的影响
    fz_mean_diff = (action_feats_a[:, 0] - action_feats_b[:, 0]).abs().mean().item()
    
    metrics = {
        'ranking_loss': loss.item(),
        'ranking_accuracy': accuracy.item(),
        'mean_prob': probs.mean().item(),
        'action_fz_diff': fz_mean_diff  # 诊断：动作差异大小
    }
    
    return loss, metrics


def generate_program_pairs(
    program_buffer: List[dict],
    reward_threshold: float = 0.1
) -> List[Tuple]:
    """从程序buffer生成训练对
    
    Args:
        program_buffer: 包含{'graph', 'reward', ...}的程序列表
        reward_threshold: 最小奖励差距（小于此值认为相等）
    
    Returns:
        pairs: [(prog_a, prog_b, preference), ...]
    """
    pairs = []
    
    # 随机采样pairs
    n = len(program_buffer)
    for _ in range(min(n * 2, 1000)):  # 生成最多1000对
        i, j = random.sample(range(n), 2)
        prog_i = program_buffer[i]
        prog_j = program_buffer[j]
        
        reward_i = prog_i.get('reward', 0.0)
        reward_j = prog_j.get('reward', 0.0)
        
        # 计算偏好
        if abs(reward_i - reward_j) < reward_threshold:
            preference = 0.5  # 相等
        elif reward_i > reward_j:
            preference = 1.0  # i更好
            pairs.append((prog_i['graph'], prog_j['graph'], preference))
        else:
            preference = 1.0  # j更好
            pairs.append((prog_j['graph'], prog_i['graph'], preference))
    
    return pairs


def integrate_ranking_value_to_mcts(
    ranking_net: RankingValueNet,
    standard_value: float,
    program_embed: torch.Tensor,
    blend_factor: float = 0.5
) -> float:
    """将ranking-based value融合到MCTS中
    
    Args:
        ranking_net: 排序网络
        standard_value: 标准value网络的估计
        program_embed: 当前程序的GNN嵌入
        blend_factor: 融合系数（0=仅用标准value，1=仅用ranking value）
    
    Returns:
        blended_value: 融合后的价值估计
    """
    with torch.no_grad():
        ranking_value = ranking_net.forward_value(program_embed).item()
    
    # 融合两个估计
    blended = (1 - blend_factor) * standard_value + blend_factor * ranking_value
    return blended


# ============================================================================
# 集成到现有训练循环的辅助函数
# ============================================================================

def setup_ranking_training(
    gnn_model,
    device: torch.device,
    learning_rate: float = 1e-4,
    embed_dim: int = 256  # GNN hidden size (默认256)
):
    """初始化ranking训练组件
    
    Args:
        gnn_model: GNN模型（用于获取嵌入维度，可选）
        device: torch设备
        learning_rate: 学习率
        embed_dim: 嵌入维度（默认256，匹配GNN v2的hidden size）
    
    Returns:
        ranking_net: RankingValueNet实例
        ranking_buffer: PairwiseRankingBuffer实例
        ranking_optimizer: torch优化器
    """
    ranking_net = RankingValueNet(embed_dim=embed_dim).to(device)
    ranking_buffer = PairwiseRankingBuffer(capacity=10000)
    ranking_optimizer = torch.optim.Adam(ranking_net.parameters(), lr=learning_rate)
    
    return ranking_net, ranking_buffer, ranking_optimizer


def train_ranking_step(
    ranking_net: RankingValueNet,
    ranking_buffer: PairwiseRankingBuffer,
    ranking_optimizer: torch.optim.Optimizer,
    gnn_encoder,
    device: torch.device,
    batch_size: int = 64
) -> Optional[dict]:
    """执行一步ranking训练
    
    Returns:
        metrics: 训练指标，如果buffer太小则返回None
    """
    if len(ranking_buffer) < batch_size:
        return None
    
    batch = ranking_buffer.sample(batch_size)
    loss, metrics = compute_ranking_loss(ranking_net, batch, gnn_encoder, device)
    
    ranking_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ranking_net.parameters(), 1.0)
    ranking_optimizer.step()
    
    return metrics


# ============================================================================
# 使用示例（集成到train_online.py）
# ============================================================================

"""
在train_online.py的Trainer类中添加：

class Trainer:
    def __init__(self, ...):
        # 现有初始化...
        
        # 添加ranking训练组件
        self.ranking_net, self.ranking_buffer, self.ranking_optimizer = \
            setup_ranking_training(self.nn_model, self.device)
        
        self.use_ranking_value = True  # 是否使用ranking value
        self.ranking_blend_factor = 0.3  # 初期保守融合
    
    def train(self):
        for iter_idx in range(self.args.total_iters):
            # MCTS搜索...
            children, visit_counts = self.mcts_search(...)
            
            # 收集程序对到ranking buffer
            for i, child_i in enumerate(children):
                for j, child_j in enumerate(children[i+1:], i+1):
                    reward_i = child_i.reward  # 假设存储了reward
                    reward_j = child_j.reward
                    
                    if abs(reward_i - reward_j) > 0.01:
                        graph_i = ast_to_pyg_graph(child_i.program)
                        graph_j = ast_to_pyg_graph(child_j.program)
                        pref = 1.0 if reward_i > reward_j else 0.0
                        self.ranking_buffer.push(graph_i, graph_j, pref)
            
            # 训练ranking网络（每次迭代）
            if len(self.ranking_buffer) >= 64:
                metrics = train_ranking_step(
                    self.ranking_net,
                    self.ranking_buffer,
                    self.ranking_optimizer,
                    self.nn_model,
                    self.device
                )
                if metrics:
                    print(f"  Ranking: loss={metrics['ranking_loss']:.4f}, "
                          f"acc={metrics['ranking_accuracy']:.2%}")
            
            # 逐步增加ranking的影响（课程学习）
            if iter_idx % 50 == 0 and self.ranking_blend_factor < 0.8:
                self.ranking_blend_factor = min(0.8, self.ranking_blend_factor + 0.1)
"""


# ============================================================================
# 动作特征提取工具
# ============================================================================

def extract_action_features_from_eval_result(eval_result: dict) -> List[float]:
    """从评估结果中提取动作幅度特征
    
    Args:
        eval_result: BatchEvaluator返回的评估结果字典
                    需要包含'action_stats'字段
    
    Returns:
        action_features: [fz_mean, fz_std, fz_max, tx_mean, tx_std, tx_max]
    
    Note:
        如果eval_result中没有action_stats，返回全零特征
    """
    if 'action_stats' not in eval_result:
        return [0.0] * 6  # 兼容旧版本
    
    stats = eval_result['action_stats']
    return [
        float(stats.get('fz_mean', 0.0)),
        float(stats.get('fz_std', 0.0)),
        float(stats.get('fz_max', 0.0)),
        float(stats.get('tx_mean', 0.0)),
        float(stats.get('tx_std', 0.0)),
        float(stats.get('tx_max', 0.0))
    ]


def compute_action_features_from_program(program, state_dict, num_samples=100):
    """直接从程序计算动作特征（用于缓存miss时）
    
    Args:
        program: 程序AST
        state_dict: 示例状态字典
        num_samples: 采样次数
    
    Returns:
        action_features: [fz_mean, fz_std, fz_max, tx_mean, tx_std, tx_max]
    """
    # 这是一个占位实现，实际应该调用evaluator
    # 在train_online.py集成时，直接使用evaluator返回的action_stats
    return [0.0] * 6
