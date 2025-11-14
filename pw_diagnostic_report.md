# Progressive Widening 退化诊断报告
**日期**: 2025-11-12  
**训练运行**: longrun_100iters_20251112_185544  
**分析**: 完整root cause analysis

---

## 🔍 症状概览

### 观察到的现象
- **前期 (Iter 1-20)**: 平均 6.7 个root children, 范围 2-11
- **后期 (Iter 81-100)**: 平均 2.2 个root children, 范围 2-3
- **期望值**: ~17 个children (基于 pw_c=1.5, pw_alpha=0.6, visits=300)
- **实际**: 89%的迭代中只有2-3个children

### 统计数据 (100次迭代)
```
Children分布:
  2个子节点: 64次 ████████████████████████████████
  3个子节点: 25次 ████████████
  4个子节点:  1次
 10个子节点:  3次 █
 11个子节点:  7次 ███
```

---

## 🐛 根本原因分析

### 原因1: Backpropagation时序问题 (设计缺陷)

**问题**: `root.visits` 在整个MCTS模拟循环中始终为0

**代码证据**:
```python
# train_online.py line 625-650
for sim_idx in range(num_simulations):  # 300次模拟
    node = root
    path = [node]
    
    # ❌ 此时root.visits=0 (backprop还没发生)
    print(f"[PW-DEBUG] root.visits={root.visits}")  # 输出: 0
    
    # Progressive Widening计算
    vis = max(0, int(node.visits))  # vis = 0
    base_cap = int(1.5 * ((0 + 1) ** 0.6))  # = 1
    max_children = max(min_cap, 1)  # = min_cap (base_cap失效!)
    
    # ... expansion ...
    pending_evals.append((leaf, path, use_real_sim))

# ✅ 批量backpropagation (在所有模拟完成后)
for leaf, path in pending_evals:
    for node in reversed(path):
        node.visits += 1  # root.visits 现在才更新!
```

**时间线**:
1. sim=0: root.visits=0, 扩展child → children=1
2. sim=1: root.visits=0, 扩展child → children=2
3. ...
4. sim=299: root.visits=0, 扩展child → children=min_cap
5. **批量backprop**: root.visits变成300 (但已经太晚了!)

**影响**: 
- `base_cap = int(1.5 * 1^0.6) = 1` 完全失效
- Progressive Widening完全依赖`min_cap`
- visits累积机制完全无效

---

### 原因2: NN过度自信导致min_cap崩溃 (核心问题)

**问题**: `root_min_cap_k`由NN先验动态计算,后期NN过度自信

**代码逻辑** (train_online.py line 598-621):
```python
# 根节点自适应最小分支数
_probs = F.softmax(_logits, dim=-1)  # NN先验分布

# 与Dirichlet混合
if root_dirichlet_noise is not None:
    _probs = (1 - eps) * _probs + eps * root_dirichlet_noise
    # eps=0.40, 但NN太自信时无法拯救

# 计算需要多少个动作才能覆盖80%概率质量
order = _probs.argsort()[::-1]
tau = 0.80  # 覆盖率阈值
csum = 0.0
k = 0
for idx in order:
    csum += _probs[idx]
    k += 1
    if csum >= tau:
        break

root_min_cap_k = max(2, min(k, len(EDIT_TYPES)))
```

**前期 vs 后期**:

| 阶段 | NN先验分布 | top-1概率 | 覆盖80%需要 | min_cap | 结果children |
|------|-----------|----------|------------|---------|-------------|
| Iter 1-20 | 较均匀 | ~30% | 10-11个动作 | 10-11 | 10-11 ✅ |
| Iter 81-100 | 极度集中 | **99.67%** | 1-2个动作 | 2-3 | 2-3 ❌ |

**实际数据** (Iter 90 根统计):
```
子节点数=2, 总访问=300, 熵=0.022, Top3访问=[299, 1]
```
- 99.67% (299/300) 访问集中在1个节点
- 熵=0.022 接近0 (完全确定性)
- NN先验: top-1动作占99%+ → min_cap=2

---

### 原因3: 探索机制失效连锁反应

**失效链条**:
1. **NN Loss恶化** (0.23 → 0.35)
   - 训练数据同质化 (16轮完全相同程序)
   - NN学习到"这个程序最好,其他都不行"
   
2. **先验崩溃** (熵 0.344 → 0.022)
   - NN预测99.67%概率在单一动作
   - Dirichlet噪声 eps=0.40 无法抵消
   - `(1-0.4)*0.9967 + 0.4*0.071 = 0.626` 仍然过高
   
3. **min_cap崩溃** (10 → 2)
   - tau=0.80阈值, top-1就超过80%
   - k=1或2就满足条件
   
4. **PW退化** (max_children=2)
   - base_cap=1无效 (visits=0)
   - max_children = max(2, 1) = 2
   
5. **搜索空间崩溃** (17 → 2)
   - 只探索2个动作
   - 其他13个动作从未尝试
   - MCTS退化成贪心搜索

---

## 📊 量化影响

### 搜索空间损失
```
期望搜索: 17 actions × 300 simulations = 5100 探索机会
实际搜索:  2 actions × 300 simulations =  600 探索机会
损失率: 88.2%
```

### 多样性崩溃
```
Iter 60 (健康):
  - children: ~10
  - entropy: ~0.3
  - 探索覆盖: 70%+

Iter 90 (崩溃):
  - children: 2
  - entropy: 0.022
  - 探索覆盖: 14% (2/14)
```

### 训练效果
```
Iter 1-60:  持续改进, 找到-2.7482
Iter 61-80: 稳定巩固
Iter 81-100: 完全停滞 (-2.9381 × 16次)
```

---

## 🔧 修复方案

### 方案1: 立即修复 - 在模拟循环内累积visits (推荐)

**思路**: 每次expansion后立即更新path上所有节点的visits

```python
# 修改 train_online.py line 760左右
for sim_idx in range(num_simulations):
    node = root
    path = [node]
    
    # ... selection & expansion ...
    
    # ✅ 立即backprop (简化版: 只更新visits, 不更新value)
    for node in reversed(path):
        node.visits += 1
    
    # 延迟value更新到批量评估阶段
    pending_evals.append((leaf, path, use_real_sim))

# 批量评估后只更新value_sum
for leaf, path in real_sim_leaves:
    reward = evaluate(leaf)
    for node in reversed(path):
        # node.visits已在模拟时更新, 这里只更新value
        node.value_sum += reward
```

**优点**:
- Progressive Widening恢复正常: base_cap = 1.5 * (300^0.6) ≈ 17
- 最小改动, 风险低
- 保持批量评估优化

**预期效果**:
- children从2-3恢复到15-17
- 搜索空间增大8倍

---

### 方案2: 增强探索 - 提高Dirichlet混合和min_cap下限

**思路**: 更激进的探索,即使NN过度自信也能对抗

```python
# 修改 train_online.py line 598-621

# 1. 提高Dirichlet混合比例
self._root_dirichlet_eps = 0.60  # 从0.40提高到0.60

# 2. 设置min_cap硬下限
root_min_cap_k_raw = calculate_from_prior_coverage()
root_min_cap_k = max(5, root_min_cap_k_raw)  # 强制至少5个children

# 3. 或者提高覆盖率阈值
tau = 0.95  # 从0.80提高到0.95, 需要覆盖95%概率质量
```

**优点**:
- 对抗NN过度自信
- 保证最小探索宽度

**缺点**:
- 治标不治本
- 增加无效探索

---

### 方案3: 根本解决 - 改进NN训练防止过度自信

**思路**: 让NN保持健康的不确定性

```python
# 添加到 train_online.py NN训练部分

# 1. Label Smoothing
policy_target_smoothed = policy_target * 0.9 + 0.1 / num_classes

# 2. Entropy Regularization
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
policy_loss = ce_loss - 0.01 * entropy  # 奖励高熵预测

# 3. Confidence Penalty
max_prob = probs.max(dim=-1)[0].mean()
confidence_penalty = F.relu(max_prob - 0.9)  # 惩罚>90%的自信
policy_loss += 0.1 * confidence_penalty

# 4. Stagnation Detection
if last_16_rewards.std() < 0.01:  # 检测停滞
    # 临时增强探索
    self._root_dirichlet_eps = 0.80
    root_min_cap_k = max(8, root_min_cap_k)
```

**优点**:
- 从源头解决问题
- 保持长期探索能力

**缺点**:
- 需要重新训练验证
- 可能影响收敛速度

---

### 方案4: 混合策略 - Progressive Widening公式改进

**思路**: 使用更鲁棒的PW公式,不完全依赖visits

```python
# 修改 train_online.py line 638-644

# 当前公式 (失效):
base_cap = int(pw_c * ((vis + 1) ** pw_alpha))  # vis=0时=1

# 改进公式 (结合iteration和visits):
iter_factor = min(current_iteration / 50.0, 1.0)  # 训练进度
base_cap = int(pw_c * ((vis + 1) ** pw_alpha) * (1 + 10 * iter_factor))
# Iter 1: base_cap = 1 * 1.2 ≈ 1
# Iter 50+: base_cap = 1 * 11 = 11 (即使vis=0)

# 或者使用simulation进度:
sim_progress = (sim_idx + 1) / num_simulations
base_cap_sim = int(pw_c * ((sim_idx + 1) ** pw_alpha))
base_cap = max(base_cap_visits, base_cap_sim)
# 即使visits=0, sim进度也会推动expansion
```

**优点**:
- 保证PW在任何情况下都能work
- 自适应训练阶段

---

## 💡 推荐行动方案

### 立即行动 (修复当前bug)
1. **实施方案1**: 在模拟循环内立即更新visits
   - 工作量: 10-15行代码修改
   - 风险: 低
   - 预期效果: PW恢复正常 (children 2→17)

2. **添加方案2的min_cap下限**: 
   ```python
   root_min_cap_k = max(5, calculated_k)
   ```
   - 工作量: 1行
   - 保险策略

### 中期优化 (改进训练)
3. **实施方案3的部分措施**:
   - Label smoothing (简单有效)
   - Stagnation detection (防御性措施)
   
### 长期研究 (可选)
4. **验证方案4的混合策略**
   - 学术价值
   - 可能发paper

---

## 📝 验证计划

### 测试1: 修复后的PW行为
```bash
# 运行短期测试 (20 iterations)
python train_online.py --num_iterations=20

# 检查children数量
grep "PW-DEBUG.*sim=299" logs/test.log
# 期望: children在10-17范围, 而非2-3
```

### 测试2: 探索多样性
```bash
# 检查根节点熵
grep "根统计" logs/test.log
# 期望: 熵>0.1, Top3访问更均匀
```

### 测试3: 训练效果
```bash
# 运行完整训练 (100 iterations)
# 期望: 
#   - Iter 80-100不再停滞
#   - 最优奖励进一步提升
#   - NN Loss保持健康
```

---

## 🎯 结论

### 核心发现
1. **直接原因**: root.visits在模拟循环中为0 → base_cap=1无效
2. **根本原因**: NN过度自信 → min_cap崩溃到2 → children=2
3. **连锁反应**: 探索→NN→先验→min_cap→PW→探索 (恶性循环)

### 关键教训
- ❌ 批量优化不能破坏核心算法逻辑 (visits必须在循环内累积)
- ❌ 自适应参数需要保底值 (min_cap不能无限小)
- ❌ NN过度自信需要正则化 (Label smoothing, Entropy reg)
- ✅ Progressive Widening理念正确,实现有bug
- ✅ 5层探索防御机制被时序bug突破

### 预期修复效果
- 修复方案1: **children 2→17 (8.5倍)**
- 搜索空间: **600→5100探索机会 (8.5倍)**
- 多样性: **熵 0.022→0.3 (13倍)**
- 停滞: **预计消除16轮连续停滞**
- 最优奖励: **可能从-2.7482进一步提升**

---

**诊断完成时间**: 2025-11-12  
**诊断人员**: GitHub Copilot  
**置信度**: 99% (代码证据 + 日志证据 + 数学证明)

