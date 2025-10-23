"""
可视化checkpoint测试结果
绘制性能曲线、规则数变化等
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT_DIR / "analysis" / "checkpoint_test_results.csv"
OUTPUT_DIR = ROOT_DIR / "analysis" / "checkpoint_figures"


def load_results() -> pd.DataFrame:
    """加载测试结果"""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"结果文件不存在: {RESULTS_FILE}\n请先运行 test_all_checkpoints.py")
    
    df = pd.read_csv(RESULTS_FILE)
    # 过滤掉无效数据
    df = df[df['verified_score'].notna()].copy()
    return df


def plot_score_evolution(df: pd.DataFrame, save_path: Path):
    """绘制得分随迭代次数的变化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 主曲线
    ax.plot(df['iteration'], df['verified_score'], 
            marker='o', linewidth=2, markersize=4, 
            label='Verified Score (test_extreme)', 
            color='#2E86AB', alpha=0.8)
    
    # 如果有train_score也画出来
    if 'train_score' in df.columns and df['train_score'].notna().any():
        ax.plot(df['iteration'], df['train_score'], 
                marker='s', linewidth=2, markersize=4, 
                label='Train Score', 
                color='#A23B72', alpha=0.6, linestyle='--')
    
    # 标注最佳点
    best_idx = df['verified_score'].idxmax()
    best_iter = df.loc[best_idx, 'iteration']
    best_score = df.loc[best_idx, 'verified_score']
    ax.scatter([best_iter], [best_score], 
              color='red', s=200, marker='*', 
              zorder=5, label=f'Best: {best_score:.4f} @ iter {best_iter}')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Checkpoint Performance Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def plot_rules_evolution(df: pd.DataFrame, save_path: Path):
    """绘制规则数随迭代的变化"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上图: 规则数变化
    ax1.plot(df['iteration'], df['num_rules'], 
            marker='o', linewidth=2, markersize=5, 
            color='#F18F01', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
    ax1.set_title('Rule Count Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 下图: 得分 vs 规则数散点图
    scatter = ax2.scatter(df['num_rules'], df['verified_score'], 
                         c=df['iteration'], cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Number of Rules', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Verified Score', fontsize=12, fontweight='bold')
    ax2.set_title('Score vs Rule Count (colored by iteration)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Iteration', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def plot_score_distribution(df: pd.DataFrame, save_path: Path):
    """绘制得分分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    ax1.hist(df['verified_score'], bins=20, 
            color='#06A77D', alpha=0.7, edgecolor='black')
    ax1.axvline(df['verified_score'].mean(), 
               color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df["verified_score"].mean():.4f}')
    ax1.axvline(df['verified_score'].median(), 
               color='blue', linestyle='--', linewidth=2, 
               label=f'Median: {df["verified_score"].median():.4f}')
    ax1.set_xlabel('Verified Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 箱线图（按规则数分组）
    rule_counts = sorted(df['num_rules'].unique())
    data_by_rules = [df[df['num_rules'] == r]['verified_score'].values 
                     for r in rule_counts]
    
    bp = ax2.boxplot(data_by_rules, labels=rule_counts, 
                     patch_artist=True, notch=True)
    
    # 美化箱线图
    for patch, color in zip(bp['boxes'], plt.cm.Set3.colors[:len(rule_counts)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Number of Rules', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Verified Score', fontsize=12, fontweight='bold')
    ax2.set_title('Score Distribution by Rule Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def plot_moving_average(df: pd.DataFrame, save_path: Path, window: int = 5):
    """绘制移动平均曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 原始曲线
    ax.plot(df['iteration'], df['verified_score'], 
            marker='o', linewidth=1, markersize=3, 
            label='Raw Score', color='gray', alpha=0.5)
    
    # 移动平均
    rolling_mean = df['verified_score'].rolling(window=window, center=True).mean()
    ax.plot(df['iteration'], rolling_mean, 
            linewidth=3, label=f'{window}-iter Moving Average', 
            color='#E63946')
    
    # 填充标准差区域
    rolling_std = df['verified_score'].rolling(window=window, center=True).std()
    ax.fill_between(df['iteration'], 
                    rolling_mean - rolling_std, 
                    rolling_mean + rolling_std, 
                    alpha=0.2, color='#E63946')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Verified Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Score Trend with {window}-iteration Moving Average', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def plot_top_checkpoints(df: pd.DataFrame, save_path: Path, top_n: int = 10):
    """绘制Top N checkpoint对比"""
    # 选择前N个最佳checkpoint
    top_df = df.nlargest(top_n, 'verified_score').sort_values('verified_score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 水平条形图
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(top_df)))
    bars = ax.barh(range(len(top_df)), top_df['verified_score'], 
                   color=colors, edgecolor='black', linewidth=1)
    
    # 设置y轴标签
    labels = [f"iter_{int(row['iteration'])} ({int(row['num_rules'])}规则)" 
              for _, row in top_df.iterrows()]
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(labels, fontsize=10)
    
    # 在条形上显示具体数值
    for i, (bar, score) in enumerate(zip(bars, top_df['verified_score'])):
        ax.text(score + 0.005, i, f'{score:.4f}', 
               va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Verified Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Checkpoints', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def create_summary_figure(df: pd.DataFrame, save_path: Path):
    """创建综合摘要图"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 得分演化（大图，占2列）
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['iteration'], df['verified_score'], 
            marker='o', linewidth=2, markersize=4, color='#2E86AB')
    best_idx = df['verified_score'].idxmax()
    ax1.scatter([df.loc[best_idx, 'iteration']], [df.loc[best_idx, 'verified_score']], 
               color='red', s=200, marker='*', zorder=5)
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Score Evolution', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 规则数分布
    ax2 = fig.add_subplot(gs[0, 2])
    rule_counts = df['num_rules'].value_counts().sort_index()
    ax2.bar(rule_counts.index, rule_counts.values, color='#F18F01', alpha=0.7)
    ax2.set_xlabel('# Rules', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Rule Distribution', fontweight='bold', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 得分vs规则数
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(df['num_rules'], df['verified_score'], 
                         c=df['iteration'], cmap='viridis', s=60, alpha=0.7)
    ax3.set_xlabel('# Rules', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Score vs Rules', fontweight='bold', fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Iteration')
    
    # 4. 得分分布直方图
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df['verified_score'], bins=15, color='#06A77D', alpha=0.7, edgecolor='black')
    ax4.axvline(df['verified_score'].mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Score', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Score Distribution', fontweight='bold', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Top 5 checkpoints
    ax5 = fig.add_subplot(gs[1, 2])
    top5 = df.nlargest(5, 'verified_score').sort_values('verified_score', ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.5, 0.9, len(top5)))
    ax5.barh(range(len(top5)), top5['verified_score'], color=colors)
    ax5.set_yticks(range(len(top5)))
    ax5.set_yticklabels([f"iter_{int(i)}" for i in top5['iteration']], fontsize=8)
    ax5.set_xlabel('Score', fontweight='bold')
    ax5.set_title('Top 5', fontweight='bold', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. 统计摘要（文本框）
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = f"""
    📊 Checkpoint测试摘要
    
    总计checkpoint数: {len(df)}
    
    得分统计:
      • 最佳得分: {df['verified_score'].max():.6f} (iter {df.loc[df['verified_score'].idxmax(), 'iteration']:.0f})
      • 平均得分: {df['verified_score'].mean():.6f}
      • 中位数: {df['verified_score'].median():.6f}
      • 标准差: {df['verified_score'].std():.6f}
    
    规则数统计:
      • 平均规则数: {df['num_rules'].mean():.1f}
      • 最佳checkpoint规则数: {df.loc[df['verified_score'].idxmax(), 'num_rules']:.0f}
      • 规则数范围: {df['num_rules'].min():.0f} - {df['num_rules'].max():.0f}
    
    迭代范围: {df['iteration'].min():.0f} - {df['iteration'].max():.0f}
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('π-Flight Checkpoint Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存综合图表: {save_path}")
    plt.close()


def main():
    """主函数：生成所有图表"""
    print("=" * 80)
    print("可视化Checkpoint测试结果")
    print("=" * 80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # 加载数据
    print(f"\n加载数据: {RESULTS_FILE}")
    df = load_results()
    print(f"✅ 加载成功，共 {len(df)} 个有效checkpoint")
    
    # 生成各种图表
    print("\n开始生成图表...")
    
    plot_score_evolution(df, OUTPUT_DIR / "01_score_evolution.png")
    plot_rules_evolution(df, OUTPUT_DIR / "02_rules_evolution.png")
    plot_score_distribution(df, OUTPUT_DIR / "03_score_distribution.png")
    plot_moving_average(df, OUTPUT_DIR / "04_moving_average.png", window=5)
    plot_top_checkpoints(df, OUTPUT_DIR / "05_top_checkpoints.png", top_n=10)
    create_summary_figure(df, OUTPUT_DIR / "00_summary.png")
    
    print("\n" + "=" * 80)
    print(f"✅ 所有图表已保存到: {OUTPUT_DIR}")
    print("=" * 80)
    
    # 显示摘要统计
    print("\n📊 数据摘要:")
    print(f"  总checkpoint数: {len(df)}")
    print(f"  最佳得分: {df['verified_score'].max():.6f} (iter {df.loc[df['verified_score'].idxmax(), 'iteration']:.0f})")
    print(f"  平均得分: {df['verified_score'].mean():.6f}")
    print(f"  得分标准差: {df['verified_score'].std():.6f}")
    print(f"  规则数范围: {df['num_rules'].min():.0f} - {df['num_rules'].max():.0f}")


if __name__ == "__main__":
    main()
