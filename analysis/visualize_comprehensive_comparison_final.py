from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(ROOT)
CSV_CANDIDATES = [
    os.path.join(WS, "results", "summaries", "comprehensive_comparison_results_collected.csv"),
    os.path.join(WS, "comprehensive_comparison_results_collected.csv"),
]
OUT_DIR = os.path.join(WS, "results", "figures")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial"]


def load_csv() -> pd.DataFrame:
    for p in CSV_CANDIDATES:
        if os.path.isfile(p):
            return pd.read_csv(p)
    raise FileNotFoundError("未找到 comprehensive_comparison_results_collected.csv，请先运行 analysis/collect_comprehensive_data.py")


def pivot_compare(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["Key"] = df2["Trajectory_Set"].astype(str) + "+" + df2["Disturbance"].astype(str)
    pvt = df2.pivot_table(index="Key", columns="Method", values="Score", aggfunc="mean")
    return pvt


def plot_bar(pvt: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)
    ax = pvt.plot(kind="bar", figsize=(12,6))
    ax.set_title("π-Flight vs CMA-ES：9 组配置对比（分数越高越好）")
    ax.set_xlabel("配置（轨迹集+扰动）")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "comprehensive_bar_comparison_final.png")
    plt.savefig(out, dpi=180)
    print(f"[SAVE] {out}")
    plt.close()


def plot_sensitivity(df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 以每个轨迹集的不同扰动为 x 轴
    traj_sets = df["Trajectory_Set"].unique().tolist()
    fig, axes = plt.subplots(1, len(traj_sets), figsize=(16,5), sharey=True)
    if len(traj_sets) == 1:
        axes = [axes]
    for i, ts in enumerate(traj_sets):
        sub = df[df["Trajectory_Set"] == ts]
        # Disturbance 顺序
        order = ["none", "mild_wind", "stress"]
        for method, g in sub.groupby("Method"):
            g2 = g.set_index("Disturbance").reindex(order).reset_index()
            axes[i].plot(order, g2["Score"], marker="o", label=method)
        axes[i].set_title(f"敏感性：{ts}")
        axes[i].set_xlabel("Disturbance")
        if i == 0:
            axes[i].set_ylabel("Score")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "disturbance_sensitivity_final.png")
    plt.savefig(out, dpi=180)
    print(f"[SAVE] {out}")
    plt.close()


def plot_heatmap(pvt: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 计算 π-Flight 相对 CMA 的优势（百分比）
    if "pi_flight" in pvt.columns and "cma_es" in pvt.columns:
        adv = (pvt["pi_flight"] - pvt["cma_es"]) / (pvt["cma_es"] + 1e-9) * 100
        # 拆分 Key 为 multi-index
        idx = adv.index.str.split("+", expand=True)
        adv_df = pd.DataFrame({"Advantage_%": adv.values}, index=pd.MultiIndex.from_frame(idx.to_frame(index=False, names=["Trajectory_Set", "Disturbance"])) )
        adv_piv = adv_df.reset_index().pivot(index="Trajectory_Set", columns="Disturbance", values="Advantage_%")
        plt.figure(figsize=(6,4))
        sns.heatmap(adv_piv, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
        plt.title("π-Flight 优势热力图（相对 CMA-ES，%）")
        out = os.path.join(OUT_DIR, "generalization_heatmap_final.png")
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        print(f"[SAVE] {out}")
        plt.close()
    else:
        print("[WARN] 缺少 pi_flight 或 cma_es 列，跳过热力图。")


def main():
    df = load_csv()
    # 类型转换
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    pvt = pivot_compare(df)
    plot_bar(pvt)
    plot_sensitivity(df)
    plot_heatmap(pvt)
    print("[DONE] 可视化已生成")


if __name__ == "__main__":
    main()
