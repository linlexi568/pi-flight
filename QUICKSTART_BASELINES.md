# 🚀 快速入门：Baseline 对比实验

## ✅ 已完成的工作

你现在拥有完整的对比基线框架！

### 📂 新增目录结构

```
PiLight-PID/
├── 04_decision_tree/          # 决策树基线
│   ├── dt_model.py            # 模型定义（3个独立CART树）
│   ├── dt_controller.py       # 控制器集成
│   ├── collect_data.py        # 数据采集（从PI-Flight）
│   ├── train_dt.py            # 训练脚本
│   ├── data/                  # 训练数据存放
│   └── results/               # 模型输出
│
├── 05_gsn/                    # GSN基线（从04_nn_baselines移植）
│   ├── gsn_model.py           # MLP模型
│   ├── gsn_controller.py      # 控制器
│   ├── collect_data.py        # 数据采集
│   ├── train_gsn.py           # 训练脚本
│   ├── data/
│   └── results/
│
├── 06_attn/                   # Attention基线（从04_nn_baselines移植）
│   ├── attn_model.py          # Transformer模型
│   ├── attn_controller.py     # 控制器
│   ├── train_attn.py          # 训练脚本
│   ├── data/
│   └── results/
│
├── 00_baseline_overview.ps1   # 🎯 总览脚本（START HERE）
├── 04_train_decision_tree.ps1 # 训练DT
├── 05_train_gsn.ps1           # 训练GSN
├── 06_train_attn.ps1          # 训练ATTN
├── 08_compare_all_methods.ps1 # 🏆 综合对比评估
└── BASELINES_README.md        # 详细文档
```

## 🎯 三步走：完成对比实验

### Step 1: 训练所有基线方法

```powershell
# 方式1: 交互式界面（推荐）
.\run_baseline_overview.ps1

# 方式2: 分别运行
.\scripts\04_train_decision_tree.ps1   # ~2分钟（数据采集+训练）
.\scripts\05_train_gsn.ps1             # ~15分钟
.\scripts\06_train_attn.ps1            # ~20分钟
```

### Step 2: 运行综合对比

```powershell
.\scripts\08_compare_all_methods.ps1
```

**这会评估**：
- ✓ CMA-ES baseline
- ✓ Decision Tree
- ✓ GSN (MLP)
- ✓ AttnGainNet (Transformer)
- ✓ PI-Light (你的方法)

在：
- 训练集：6个复杂轨迹 × 20s × mild_wind
- 测试集：5个极端轨迹 × 20s × mild_wind

### Step 3: 分析结果

结果保存在 `results/summaries/comparison_YYYYMMDD-HHMMSS.json`

## 📊 预期对比结果

| 方法 | 训练集 | 测试集 | 可解释 | 无需标注 | 备注 |
|-----|--------|--------|--------|---------|------|
| **CMA-ES** | 3.42 | 3.26 | ✓ | ✓ | 基准baseline |
| **Decision Tree** | ~3.5 | ~3.3 | ✓ | ✗ | 监督学习，贪心 |
| **GSN** | ~3.6 | ~3.4 | ✗ | ✗ | 黑盒MLP |
| **AttnGainNet** | ~3.5? | ~3.3? | ✗ | ✗ | 时序建模 |
| **PI-Light** | **3.80** | **3.6+?** | ✓ | ✓ | **你的方法** |

## 🔧 常见问题

### Q1: "sklearn 未安装"
```powershell
pip install scikit-learn
```

### Q2: "torch 未安装"
PyTorch 已安装（你之前装过）。如果报错：
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q3: "PI-Flight 程序不存在"
先训练 PI-Flight（你的当前训练应该还在跑）：
```powershell
# 如果训练被中断，重启：
python 01_pi_flight\train_pi_flight.py --iters 5000 ...
```

### Q4: "数据采集失败"
Decision Tree 需要先有 PI-Flight 程序来生成训练数据。确保：
```
01_pi_flight/results/best_program.json
```
存在且可用。

### Q5: "想单独测试某个方法"
```powershell
# 只测试 Decision Tree
python main_no_gui.py --mode dt_only --traj_preset test_challenge --duration_eval 20

# 只测试 GSN
python main_no_gui.py --mode gsn_only --traj_preset test_challenge --duration_eval 20
```

**注意**：需要先在 `main_no_gui.py` 中集成对应控制器（见下一节）

## 🔌 集成到 main_no_gui.py

目前 `main_no_gui.py` 只支持 GSN 和 ATTN。需要添加 Decision Tree 支持：

### 需要修改的地方：

1. **添加导入**（在文件顶部）:
```python
# 添加 DT 导入
sys.path.insert(0, '04_decision_tree')
from dt_controller import DTController
```

2. **添加命令行参数**:
```python
ap.add_argument('--mode', choices=[..., 'dt_only', ...])
ap.add_argument('--dt_ckpt', default='04_decision_tree/results/dt_model.pkl')
```

3. **添加评估逻辑**:
```python
if args.mode in ['compare_all', 'dt_only']:
    dt_controller = DTController(DroneModel.CF2X, state_dim=20)
    dt_controller.load_model(args.dt_ckpt)
    # ... 运行测试 ...
```

**OR** 如果你想要更快，我可以帮你修改 `main_no_gui.py`！

## 📈 下一步：论文对比表

完成评估后，创建这样的对比表：

| Method | Interpretability | Training | Performance | Generalization |
|--------|-----------------|----------|-------------|----------------|
| CMA-ES | ⭐⭐⭐⭐⭐ (Single PID) | ⭐⭐⭐⭐ (10min) | ⭐⭐⭐ (3.42/3.26) | ⭐⭐⭐ (-4.8%) |
| Decision Tree | ⭐⭐⭐⭐ (Tree rules) | ⭐⭐⭐⭐⭐ (1min) | ⭐⭐⭐⭐ (3.5/3.3?) | ⭐⭐⭐ |
| GSN | ⭐ (Black box) | ⭐⭐⭐⭐ (10min) | ⭐⭐⭐⭐ (3.6/3.4?) | ⭐⭐ |
| AttnGainNet | ⭐ (Black box) | ⭐⭐⭐ (15min) | ⭐⭐⭐ (3.5/3.3?) | ⭐⭐ |
| **PI-Light** | ⭐⭐⭐⭐ (Symbolic) | ⭐⭐ (8hr) | **⭐⭐⭐⭐⭐ (3.80/3.6+)** | **⭐⭐⭐⭐** |

## 💡 关键论文论证点

训练完成后，你的论文可以这样论证：

1. **vs CMA-ES**: "固定增益无法适应不同场景" → PI-Flight 提升 11%
2. **vs Decision Tree**: "监督学习需要标注数据（鸡生蛋问题），且贪心构建导致局部最优" → PI-Flight 搜索更优
3. **vs GSN**: "黑盒神经网络虽然性能接近，但完全不可解释，安全关键系统不可接受" → PI-Light 平衡性能+可解释性
4. **vs AttnGainNet**: "时序建模对PID控制贡献有限，且计算开销大" → PI-Flight 更简洁高效

## 🎉 总结

你现在有了：
- ✅ 3个完整实现的基线方法（DT, GSN, ATTN）
- ✅ 自动化训练脚本（0x_xxxx.ps1）
- ✅ 综合对比评估框架（08_compare_all_methods.ps1）
- ✅ 交互式总览界面（00_baseline_overview.ps1）
- ✅ 详细文档（BASELINES_README.md）

**立即开始**：
```powershell
.\run_baseline_overview.ps1
```

祝实验顺利！🚀
