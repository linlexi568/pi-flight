"""
对比MCTS和CMA-ES在test_challenge预设下的性能
使用统一的测试配置
"""
import subprocess
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

ROOT_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
VERIFY_SCRIPT = ROOT_DIR / "utilities" / "verify_program.py"

# 测试配置（与checkpoint测试一致：test_extreme）
TEST_CONFIG = {
    'traj_preset': 'test_extreme',
    'aggregate': 'harmonic',
    'disturbance': 'stress',
    'duration': '25',
    'log_skip': '2',
    'clip_D': '1.2',
}


def test_program(program_path: Path, label: str) -> dict:
    """测试单个程序"""
    print(f"\n{'='*80}")
    print(f"🔍 测试 {label}")
    print(f"{'='*80}")
    print(f"程序路径: {program_path}")
    
    cmd = [
        str(VENV_PYTHON),
        str(VERIFY_SCRIPT),
        "--program", str(program_path),
        "--traj_preset", TEST_CONFIG['traj_preset'],
        "--aggregate", TEST_CONFIG['aggregate'],
        "--disturbance", TEST_CONFIG['disturbance'],
        "--duration", TEST_CONFIG['duration'],
        "--log-skip", TEST_CONFIG['log_skip'],
        "--clip-D", TEST_CONFIG['clip_D'],
        "--compose-by-gain",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
        output = result.stdout
        
        # 解析得分
        score = None
        per_traj = {}
        
        for line in output.split('\n'):
            if '[Verified]' in line and '聚合得分:' in line:
                parts = line.split('聚合得分:')
                if len(parts) > 1:
                    score_str = parts[1].strip().split()[0]
                    score = float(score_str)
            
            if '逐轨迹:' in line:
                try:
                    import re
                    match = re.search(r'\[({.*})\]', line.replace('\n', ''))
                    if match:
                        import ast
                        traj_list = ast.literal_eval('[' + match.group(1) + ']')
                        for item in traj_list:
                            per_traj[item['traj']] = item['reward']
                except:
                    pass
        
        if score is None:
            print(f"⚠️ 警告: 无法解析得分")
            print(f"输出前500字符:\n{output[:500]}")
            return None
        
        print(f"✅ 得分: {score:.6f}")
        
        # 读取程序信息
        with open(program_path, 'r', encoding='utf-8') as f:
            prog_data = json.load(f)
        num_rules = len(prog_data.get('rules', []))
        
        return {
            'label': label,
            'score': score,
            'num_rules': num_rules,
            'per_traj': per_traj,
            'program_path': str(program_path)
        }
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None


def load_checkpoint_results() -> pd.DataFrame:
    """加载checkpoint测试结果"""
    csv_path = ROOT_DIR / "analysis" / "checkpoint_test_results.csv"
    if not csv_path.exists():
        print(f"⚠️ 找不到checkpoint测试结果: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df = df[df['verified_score'].notna()].copy()
    return df


def visualize_comparison(mcts_result: dict, cma_result: dict, checkpoint_df: pd.DataFrame = None):
    """可视化对比结果"""
    output_dir = ROOT_DIR / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 图1: 总体对比条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['MCTS\n(Best)', 'CMA-ES\n(Baseline)']
    scores = [mcts_result['score'], cma_result['score']]
    colors = ['#2E86AB', '#E63946']
    
    bars = ax.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 添加提升百分比
    improvement = ((mcts_result['score'] - cma_result['score']) / cma_result['score']) * 100
    ax.text(0.5, max(scores) * 0.95, 
            f'MCTS Advantage: +{improvement:.2f}%',
            transform=ax.transData,
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_ylabel('Aggregated Score (Harmonic Mean)', fontsize=12, fontweight='bold')
    ax.set_title('MCTS vs CMA-ES Performance Comparison\n(test_extreme + stress disturbances + 25s)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([min(scores) * 0.95, max(scores) * 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_overall_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {output_dir / '01_overall_comparison.png'}")
    plt.close()
    
    # 图2: 逐轨迹对比
    if mcts_result['per_traj'] and cma_result['per_traj']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        trajectories = sorted(set(mcts_result['per_traj'].keys()) | set(cma_result['per_traj'].keys()))
        x = np.arange(len(trajectories))
        width = 0.35
        
        mcts_scores = [mcts_result['per_traj'].get(t, 0) for t in trajectories]
        cma_scores = [cma_result['per_traj'].get(t, 0) for t in trajectories]
        
        bars1 = ax.bar(x - width/2, mcts_scores, width, label='MCTS', 
                      color='#2E86AB', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, cma_scores, width, label='CMA-ES', 
                      color='#E63946', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Trajectory Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(trajectories, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "02_per_trajectory_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved figure: {output_dir / '02_per_trajectory_comparison.png'}")
        plt.close()
    
    # 图3: 如果有checkpoint数据，显示演化曲线（分数和规则数，双子图）
    if checkpoint_df is not None and len(checkpoint_df) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 子图1: 分数演化
        ax1.plot(checkpoint_df['iteration'], checkpoint_df['verified_score'],
               marker='o', linewidth=2, markersize=5, 
               label='MCTS Training Process', color='#0C58FD', alpha=0.8)
        
        # 标注最佳点
        best_idx = checkpoint_df['verified_score'].idxmax()
        best_iter = checkpoint_df.loc[best_idx, 'iteration']
        best_score = checkpoint_df.loc[best_idx, 'verified_score']
        ax1.scatter([best_iter], [best_score], 
                  color='red', s=200, marker='*', zorder=5,
                  label=f'MCTS Best: {best_score:.4f}')
        
        # CMA-ES基线
        ax1.axhline(y=cma_result['score'], color='#E63946', 
                  linestyle='--', linewidth=2, 
                  label=f'CMA-ES Baseline: {cma_result["score"]:.4f}')
        
        ax1.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Validation Score', fontsize=12, fontweight='bold')
        ax1.set_title('MCTS Training Evolution vs CMA-ES Baseline', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 规则数演化
        ax2.plot(checkpoint_df['iteration'], checkpoint_df['num_rules'],
               marker='s', linewidth=2, markersize=5,
               label='Number of Rules', color="#06A77D", alpha=0.8)
        
        # CMA-ES规则数基线
        ax2.axhline(y=cma_result['num_rules'], color='#E63946',
                  linestyle='--', linewidth=2,
                  label=f'CMA-ES Rules: {cma_result["num_rules"]}')
        
        ax2.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
        ax2.set_title('Controller Complexity Evolution', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, checkpoint_df['num_rules'].max() + 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / "03_training_evolution.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved figure: {output_dir / '03_training_evolution.png'}")
        plt.close()
    
    # 图4: 规则数对比
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['MCTS', 'CMA-ES']
    rules = [mcts_result['num_rules'], cma_result['num_rules']]
    
    bars = ax.bar(methods, rules, color=['#2E86AB', '#E63946'], alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, rule_count in zip(bars, rules):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(rule_count)} Rules',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
    ax.set_title('Controller Complexity Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_complexity_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {output_dir / '04_complexity_comparison.png'}")
    plt.close()
    
    print(f"\n✅ All figures saved to: {output_dir}")


def main():
    print("=" * 80)
    print("🎯 MCTS vs CMA-ES Performance Comparison")
    print("=" * 80)
    print(f"\nTest Configuration:")
    for key, value in TEST_CONFIG.items():
        print(f"  • {key}: {value}")
    
    # 测试MCTS最佳checkpoint（从checkpoint测试中找到）
    checkpoint_df = load_checkpoint_results()
    best_iter = None
    if checkpoint_df is not None and len(checkpoint_df) > 0:
        best_idx = checkpoint_df['verified_score'].idxmax()
        best_iter = checkpoint_df.loc[best_idx, 'iteration']
        mcts_path = ROOT_DIR / "01_pi_flight" / "results" / "checkpoints" / f"best_program_iter_{int(best_iter):06d}.json"
        print(f"\n📍 Using MCTS best checkpoint: iter_{int(best_iter)} (score: {checkpoint_df.loc[best_idx, 'verified_score']:.6f})")
    else:
        mcts_path = ROOT_DIR / "01_pi_flight" / "results" / "best_program.json"
        print(f"\n⚠️ No checkpoint data, using default best_program.json")
    
    label = f"MCTS (Best: iter_{int(best_iter) if best_iter is not None else 'N/A'})"
    mcts_result = test_program(mcts_path, label)
    
    # 测试CMA-ES基线
    cma_path = ROOT_DIR / "03_CMA-ES" / "results" / "best_program.json"
    cma_result = test_program(cma_path, "CMA-ES (Baseline)")
    
    if not mcts_result or not cma_result:
        print("\n❌ Testing failed, cannot generate comparison")
        return
    
    # 加载checkpoint结果
    checkpoint_df = load_checkpoint_results()
    
    # 生成对比可视化
    print(f"\n{'='*80}")
    print("📊 Generating visualization charts...")
    print(f"{'='*80}")
    visualize_comparison(mcts_result, cma_result, checkpoint_df)
    
    # 打印摘要
    print(f"\n{'='*80}")
    print("📈 Comparison Summary")
    print(f"{'='*80}")
    print(f"\nMCTS:")
    print(f"  • Score: {mcts_result['score']:.6f}")
    print(f"  • Number of Rules: {mcts_result['num_rules']}")
    print(f"\nCMA-ES:")
    print(f"  • Score: {cma_result['score']:.6f}")
    print(f"  • Number of Rules: {cma_result['num_rules']}")
    
    improvement = ((mcts_result['score'] - cma_result['score']) / cma_result['score']) * 100
    print(f"\nPerformance Improvement: {improvement:+.2f}%")
    
    if checkpoint_df is not None:
        print(f"\nCheckpoint Statistics:")
        print(f"  • Total: {len(checkpoint_df)}")
        print(f"  • Best: {checkpoint_df['verified_score'].max():.6f} @ iter {checkpoint_df.loc[checkpoint_df['verified_score'].idxmax(), 'iteration']:.0f}")
        print(f"  • Average: {checkpoint_df['verified_score'].mean():.6f}")
        print(f"  • 平均: {checkpoint_df['verified_score'].mean():.6f}")


if __name__ == "__main__":
    main()
