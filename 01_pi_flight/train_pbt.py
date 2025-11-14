#!/usr/bin/env python3
"""
Population-Based Training (PBT) for MCTS-based Program Synthesis
=================================================================

16-agent并行训练，自动调节MCTS超参数：
- 每个agent独立训练GNN+MCTS
- 周期性评估性能，淘汰弱agent
- 复制强agent的权重+扰动参数
- 共享Isaac Gym环境池（512环境）

参考文献:
- PBT: Jaderberg et al. (2018) "Population Based Training of Neural Networks"
- AlphaZero: Silver et al. (2017) "Mastering Chess and Shogi by Self-Play"
"""

import argparse
import time
import json
import random
import copy
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 导入基础训练模块
from train_online import (
    OnlineTrainer, ReplayBuffer, 
    EDIT_TYPES, ast_to_pyg_graph,
    save_program_json, get_program_hash
)

# GNN模块
from models.gnn_policy_nn_v2 import create_gnn_policy_value_net_v2
from torch_geometric.data import Batch as PyGBatch

# Evaluator
from utils.batch_evaluation import BatchEvaluator


class PBTAgent:
    """单个PBT agent（包含GNN模型、MCTS参数、训练状态）"""
    
    def __init__(self, agent_id: int, args, device, shared_encoder=None):
        self.id = agent_id
        self.args = args
        self.device = device
        
        # MCTS参数（可演化的）
        self.mcts_params = self._initialize_mcts_params()
        
        # 训练超参（可演化的）
        self.learning_rate = 10 ** np.random.uniform(-4, -2.5)  # 1e-4 到 3e-3
        
        # GNN模型（如果使用共享编码器）
        self.shared_encoder = shared_encoder
        if shared_encoder is not None:
            # 只创建独立的policy head
            self.policy_head = self._create_policy_head()
            self.nn_model = None  # 标记使用共享模式
        else:
            # 创建完整的独立模型
            self.nn_model = create_gnn_policy_value_net_v2(
                node_feat_dim=args.node_feat_dim if hasattr(args, 'node_feat_dim') else 16,
                hidden_channels=args.hidden_channels if hasattr(args, 'hidden_channels') else 128,
                num_gnn_layers=args.num_gnn_layers if hasattr(args, 'num_gnn_layers') else 3,
                n_edit_types=len(EDIT_TYPES),
                dropout=0.1
            ).to(device)
            self.policy_head = None
        
        # 优化器
        self._setup_optimizer()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=args.replay_capacity)
        
        # 性能追踪
        self.performance_history = deque(maxlen=20)  # 最近20轮
        self.best_reward = -float('inf')
        self.best_program = None
        self.total_iterations = 0
        
        # 从OnlineTrainer继承的训练器（复用MCTS逻辑）
        self.trainer = None  # 稍后初始化
        
        print(f"[Agent {self.id}] 初始化完成，MCTS参数: {self.mcts_params}")
    
    def _initialize_mcts_params(self) -> Dict[str, float]:
        """随机初始化MCTS参数"""
        return {
            'puct_c': np.random.uniform(1.0, 2.5),
            'exploration_weight': np.random.uniform(1.5, 4.0),
            'dirichlet_eps': np.random.uniform(0.15, 0.45),
            'dirichlet_alpha': np.random.uniform(0.2, 0.5),
            'temperature': np.random.uniform(0.6, 1.8),
            'simulations': int(np.random.choice([400, 600, 800]))  # 降低以支持16 agents
        }
    
    def _create_policy_head(self) -> nn.Module:
        """创建独立的policy head（用于共享编码器模式）"""
        hidden_dim = self.args.hidden_channels if hasattr(self.args, 'hidden_channels') else 128
        
        head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(EDIT_TYPES))
        ).to(self.device)
        
        return head
    
    def _setup_optimizer(self):
        """设置优化器"""
        if self.shared_encoder is not None:
            # 共享模式：优化shared_encoder + policy_head
            params = list(self.shared_encoder.parameters()) + list(self.policy_head.parameters())
        else:
            # 独立模式：优化整个nn_model
            params = self.nn_model.parameters()
        
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
    
    def forward(self, graph):
        """前向传播"""
        if self.shared_encoder is not None:
            # 共享编码器模式
            embedding = self.shared_encoder.get_embedding(graph)
            policy_logits = self.policy_head(embedding)
            return policy_logits, None, None  # 返回格式兼容
        else:
            # 独立模型模式
            return self.nn_model(graph)
    
    def get_model_state_dict(self):
        """获取模型状态字典"""
        if self.shared_encoder is not None:
            return {
                'shared_encoder': self.shared_encoder.state_dict(),
                'policy_head': self.policy_head.state_dict()
            }
        else:
            return {'nn_model': self.nn_model.state_dict()}
    
    def load_model_state_dict(self, state_dict):
        """加载模型状态字典"""
        if self.shared_encoder is not None:
            self.shared_encoder.load_state_dict(state_dict['shared_encoder'])
            self.policy_head.load_state_dict(state_dict['policy_head'])
        else:
            self.nn_model.load_state_dict(state_dict['nn_model'])
    
    def copy_from(self, other_agent: 'PBTAgent'):
        """从另一个agent复制权重"""
        self.load_model_state_dict(other_agent.get_model_state_dict())
        print(f"[Agent {self.id}] 复制 Agent {other_agent.id} 的权重")
    
    def perturb_params(self, perturb_factors=(0.8, 1.2)):
        """扰动MCTS参数"""
        for key, value in self.mcts_params.items():
            if key == 'simulations':
                # simulations离散选择
                continue
            factor = random.choice(perturb_factors)
            new_value = value * factor
            
            # 约束到合理范围
            if key == 'puct_c':
                new_value = np.clip(new_value, 0.5, 3.0)
            elif key == 'exploration_weight':
                new_value = np.clip(new_value, 1.0, 5.0)
            elif key == 'dirichlet_eps':
                new_value = np.clip(new_value, 0.05, 0.6)
            elif key == 'dirichlet_alpha':
                new_value = np.clip(new_value, 0.1, 0.7)
            elif key == 'temperature':
                new_value = np.clip(new_value, 0.3, 2.5)
            
            self.mcts_params[key] = new_value
        
        # 学习率也扰动
        self.learning_rate *= random.choice(perturb_factors)
        self.learning_rate = np.clip(self.learning_rate, 1e-5, 1e-2)
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        
        print(f"[Agent {self.id}] 参数扰动完成: {self.mcts_params}, lr={self.learning_rate:.2e}")


class PBTTrainer:
    """Population-Based Training 主训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"Population-Based Training for MCTS Program Synthesis")
        print(f"{'='*60}")
        print(f"Agent数量: {args.n_agents}")
        print(f"PBT间隔: 每{args.pbt_interval}轮评估一次")
        print(f"淘汰率: {args.exploit_threshold*100:.0f}%")
        print(f"设备: {self.device}")
        print(f"{'='*60}\n")
        
        # 初始化evaluator（所有agent共享）
        self.evaluator = BatchEvaluator(
            trajectory_type=args.traj,
            duration=args.duration,
            num_envs=args.isaac_num_envs,
            device='cuda:0',
            headless=True,
            reward_profile=args.reward_profile
        )
        
        # 创建共享的GNN编码器（可选，节省显存）
        if args.shared_encoder:
            print("[PBT] 使用共享GNN编码器模式（节省显存）")
            self.shared_encoder = self._create_shared_encoder()
        else:
            print("[PBT] 使用独立GNN模型模式")
            self.shared_encoder = None
        
        # 初始化所有agents
        self.agents: List[PBTAgent] = []
        for i in range(args.n_agents):
            agent = PBTAgent(i, args, self.device, self.shared_encoder)
            self.agents.append(agent)
        
        # PBT统计
        self.global_best_reward = -float('inf')
        self.global_best_program = None
        self.global_best_agent_id = -1
        
        # 每个agent的训练器（复用OnlineTrainer的MCTS逻辑）
        self._setup_agent_trainers()
    
    def _create_shared_encoder(self):
        """创建共享的GNN编码器"""
        model = create_gnn_policy_value_net_v2(
            node_feat_dim=self.args.node_feat_dim if hasattr(self.args, 'node_feat_dim') else 16,
            hidden_channels=self.args.hidden_channels if hasattr(self.args, 'hidden_channels') else 128,
            num_gnn_layers=self.args.num_gnn_layers if hasattr(self.args, 'num_gnn_layers') else 3,
            n_edit_types=len(EDIT_TYPES),
            dropout=0.1
        ).to(self.device)
        
        return model.gnn_encoder  # 只返回编码器部分
    
    def _setup_agent_trainers(self):
        """为每个agent创建训练器实例（复用OnlineTrainer的MCTS逻辑）"""
        for agent in self.agents:
            # 创建一个轻量级的trainer wrapper
            # 这里我们复用OnlineTrainer的mcts_search方法
            # 但使用agent自己的参数
            agent.trainer = self._create_agent_trainer_wrapper(agent)
    
    def _create_agent_trainer_wrapper(self, agent: PBTAgent):
        """创建agent的训练器wrapper"""
        # 这是一个简化的wrapper，复用OnlineTrainer的核心方法
        class AgentTrainerWrapper:
            def __init__(self, parent_trainer, agent):
                self.parent = parent_trainer
                self.agent = agent
                self.args = agent.args
                self.device = agent.device
                self.evaluator = parent_trainer.evaluator
                
                # 使用agent的模型
                if agent.nn_model is not None:
                    self.nn_model = agent.nn_model
                else:
                    # 共享编码器模式：创建临时的forward wrapper
                    self.nn_model = lambda graph: agent.forward(graph)
                
                # 使用agent的MCTS参数
                self._update_mcts_params()
            
            def _update_mcts_params(self):
                """从agent同步MCTS参数"""
                params = self.agent.mcts_params
                self._puct_c = params['puct_c']
                self._exploration_weight = params['exploration_weight']
                self._root_dirichlet_eps = params['dirichlet_eps']
                self._root_dirichlet_alpha = params['dirichlet_alpha']
                self._policy_temperature = params['temperature']
                self._max_depth = 12  # 固定
            
            def mcts_search(self, *args, **kwargs):
                """调用父trainer的mcts_search（但使用agent的参数）"""
                # 这里需要复用OnlineTrainer.mcts_search的完整逻辑
                # 为了简化，我们暂时标记为TODO
                # 实际实现中需要完整复制或重构mcts_search
                pass  # TODO: 实现
        
        return AgentTrainerWrapper(self, agent)
    
    def train(self):
        """PBT主训练循环"""
        for global_iter in range(self.args.total_iters):
            iter_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"Iteration {global_iter + 1} / {self.args.total_iters}")
            print(f"{'='*60}")
            
            # 1. 所有agent并行训练一步
            agent_rewards = self._parallel_train_step(global_iter)
            
            # 2. 更新全局最佳
            self._update_global_best(agent_rewards, global_iter)
            
            # 3. PBT调度（周期性）
            if (global_iter + 1) % self.args.pbt_interval == 0:
                self._pbt_exploit_explore()
            
            # 4. 统计输出
            iter_time = time.time() - iter_start
            self._print_statistics(global_iter, agent_rewards, iter_time)
            
            # 5. 保存checkpoint
            if (global_iter + 1) % self.args.save_freq == 0:
                self._save_checkpoint(global_iter)
        
        # 最终保存
        self._save_final_results()
    
    def _parallel_train_step(self, iteration: int) -> List[float]:
        """所有agent并行训练一步，返回每个agent的奖励"""
        agent_rewards = []
        
        # TODO: 这里需要实现每个agent的完整训练步骤
        # 包括：MCTS搜索 -> 生成程序 -> 评估 -> 更新NN
        
        # 简化示例（需要完整实现）
        for agent in self.agents:
            # 1. 运行MCTS（使用agent的参数）
            # next_program = agent.trainer.mcts_search(...)
            
            # 2. 评估程序
            # reward = self.evaluator.evaluate_single(next_program)
            
            # 3. 添加到replay buffer
            # agent.replay_buffer.push(sample)
            
            # 4. 更新NN（每N轮）
            # if iteration % agent.args.update_freq == 0:
            #     agent_train_step(agent)
            
            # 临时：随机奖励（测试PBT逻辑）
            reward = np.random.uniform(-10, 0)
            agent.performance_history.append(reward)
            if reward > agent.best_reward:
                agent.best_reward = reward
            
            agent_rewards.append(reward)
        
        return agent_rewards
    
    def _update_global_best(self, agent_rewards: List[float], iteration: int):
        """更新全局最佳agent"""
        for i, (agent, reward) in enumerate(zip(self.agents, agent_rewards)):
            if reward > self.global_best_reward:
                self.global_best_reward = reward
                self.global_best_program = agent.best_program
                self.global_best_agent_id = agent.id
                print(f"[Iter {iteration+1}] 🎉 新全局最佳！Agent {agent.id}, Reward: {reward:.4f}")
    
    def _pbt_exploit_explore(self):
        """PBT的核心：Exploit & Explore"""
        print(f"\n{'='*60}")
        print(f"PBT调度：Exploit & Explore")
        print(f"{'='*60}")
        
        # 1. 计算每个agent的性能（最近10轮平均）
        performances = []
        for agent in self.agents:
            if len(agent.performance_history) > 0:
                recent = list(agent.performance_history)[-10:]
                perf = np.mean(recent)
            else:
                perf = -float('inf')
            performances.append((agent.id, perf, agent))
        
        # 2. 排序
        performances.sort(key=lambda x: x[1], reverse=True)
        
        # 打印排名
        print("\nAgent性能排名:")
        for rank, (agent_id, perf, agent) in enumerate(performances, 1):
            marker = "🏆" if rank <= 3 else ("⭐" if rank <= len(self.agents)//2 else "")
            print(f"  {rank}. Agent {agent_id}: {perf:.4f} {marker}")
        
        # 3. 淘汰下位20%，复制上位agent
        n_exploit = max(1, int(self.args.n_agents * self.args.exploit_threshold))
        
        top_agents = [agent for _, _, agent in performances[:n_exploit]]
        bottom_agents = [agent for _, _, agent in performances[-n_exploit:]]
        
        print(f"\n淘汰下位{n_exploit}个agent，复制上位agent:")
        for weak_agent in bottom_agents:
            # 随机选一个强agent
            strong_agent = random.choice(top_agents)
            
            print(f"  🔄 Agent {weak_agent.id} (⭐{weak_agent.best_reward:.2f}) "
                  f"复制 Agent {strong_agent.id} (⭐{strong_agent.best_reward:.2f})")
            
            # 复制权重
            weak_agent.copy_from(strong_agent)
            
            # 扰动参数
            weak_agent.perturb_params(perturb_factors=(0.8, 1.2))
        
        print(f"{'='*60}\n")
    
    def _print_statistics(self, iteration: int, agent_rewards: List[float], iter_time: float):
        """打印统计信息"""
        mean_reward = np.mean(agent_rewards)
        max_reward = np.max(agent_rewards)
        min_reward = np.min(agent_rewards)
        std_reward = np.std(agent_rewards)
        
        print(f"\n统计:")
        print(f"  平均奖励: {mean_reward:.4f}")
        print(f"  最大奖励: {max_reward:.4f}")
        print(f"  最小奖励: {min_reward:.4f}")
        print(f"  标准差: {std_reward:.4f}")
        print(f"  全局最佳: {self.global_best_reward:.4f} (Agent {self.global_best_agent_id})")
        print(f"  用时: {iter_time:.2f}s")
    
    def _save_checkpoint(self, iteration: int):
        """保存checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'global_best_reward': self.global_best_reward,
            'global_best_agent_id': self.global_best_agent_id,
            'agents': []
        }
        
        for agent in self.agents:
            agent_data = {
                'id': agent.id,
                'mcts_params': agent.mcts_params,
                'learning_rate': agent.learning_rate,
                'best_reward': agent.best_reward,
                'model_state': agent.get_model_state_dict()
            }
            checkpoint['agents'].append(agent_data)
        
        save_path = self.args.save_path.replace('.json', f'_pbt_iter{iteration+1}.pt')
        torch.save(checkpoint, save_path)
        print(f"  💾 Checkpoint已保存: {save_path}")
    
    def _save_final_results(self):
        """保存最终结果"""
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"全局最佳奖励: {self.global_best_reward:.4f}")
        print(f"来自Agent {self.global_best_agent_id}")
        print(f"{'='*60}\n")
        
        # 保存最佳程序
        if self.global_best_program is not None:
            final_path = self.args.save_path.replace('.json', '_pbt_final.json')
            save_program_json(self.global_best_program, final_path)
            print(f"最佳程序已保存: {final_path}")


def parse_args():
    p = argparse.ArgumentParser(description='Population-Based Training for MCTS Program Synthesis')
    
    # PBT参数
    p.add_argument('--n-agents', type=int, default=16, help='Agent数量')
    p.add_argument('--pbt-interval', type=int, default=50, help='PBT调度间隔（轮数）')
    p.add_argument('--exploit-threshold', type=float, default=0.25, help='淘汰比例（0-1）')
    p.add_argument('--shared-encoder', action='store_true', help='使用共享GNN编码器（节省显存）')
    
    # 训练参数
    p.add_argument('--total-iters', type=int, default=5000, help='总迭代数')
    p.add_argument('--update-freq', type=int, default=50, help='NN更新频率')
    p.add_argument('--train-steps-per-update', type=int, default=10, help='每次更新的训练步数')
    p.add_argument('--batch-size', type=int, default=64, help='批量大小（降低以适应多agent）')
    p.add_argument('--replay-capacity', type=int, default=20000, help='每个agent的replay buffer容量')
    
    # GNN参数
    p.add_argument('--node-feat-dim', type=int, default=16, help='节点特征维度')
    p.add_argument('--hidden-channels', type=int, default=128, help='隐藏层维度')
    p.add_argument('--num-gnn-layers', type=int, default=3, help='GNN层数')
    p.add_argument('--learning-rate', type=float, default=1e-3, help='初始学习率（PBT会调整）')
    
    # 仿真参数
    p.add_argument('--traj', type=str, default='figure8', choices=['hover', 'figure8', 'circle', 'helix'])
    p.add_argument('--duration', type=int, default=10, help='仿真时长（秒）')
    p.add_argument('--isaac-num-envs', type=int, default=512, help='Isaac Gym并行环境数')
    p.add_argument('--reward-profile', type=str, default='control_law_discovery', 
                   choices=['default', 'control_law_discovery', 'smooth_control', 'balanced_smooth'])
    
    # 保存参数
    p.add_argument('--save-path', type=str, default='results/pbt_best_program.json')
    p.add_argument('--save-freq', type=int, default=200, help='Checkpoint保存频率')
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 创建PBT训练器
    pbt_trainer = PBTTrainer(args)
    
    # 开始训练
    pbt_trainer.train()
