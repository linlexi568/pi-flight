from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(ROOT)
CSV_CANDIDATES = [
    os.path.join(WS, "results", "summaries", "comprehensive_comparison_results_collected.csv"),
    os.path.join(WS, "comprehensive_comparison_results_collected.csv"),
]
OUT_DIR = os.path.join(WS, "results", "summaries")
OUT_MD = os.path.join(OUT_DIR, "comparison_depth_summary.md")


def load_csv() -> pd.DataFrame:
    for p in CSV_CANDIDATES:
        if os.path.isfile(p):
            return pd.read_csv(p)
    raise FileNotFoundError("未找到 comprehensive_comparison_results_collected.csv，请先运行 analysis/collect_comprehensive_data.py")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_csv()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    # 仅保留 pi_flight 与 cma_es 两类
    methods = set(df["Method"].unique().tolist())
    if not ({"pi_flight", "cma_es"} <= methods):
        raise RuntimeError(f"缺少方法列，现有: {methods}")

    # 每个配置（Traj_Set×Disturbance）求相对优势
    key = df["Trajectory_Set"].astype(str) + "+" + df["Disturbance"].astype(str)
    df = df.assign(Key=key)
    pvt = df.pivot_table(index="Key", columns="Method", values="Score", aggfunc="mean")
    pvt = pvt.dropna()
    pvt["Advantage_%"] = (pvt["pi_flight"] - pvt["cma_es"]) / (pvt["cma_es"] + 1e-9) * 100

    avg_adv = pvt["Advantage_%"].mean()
    wins = int((pvt["Advantage_%"] > 0).sum())
    total = int(len(pvt))

    lines = []
    lines.append("# 深入比较统计摘要\n")
    lines.append(f"- 配置总数: {total}")
    lines.append(f"- π-Flight 胜场: {wins}/{total}")
    lines.append(f"- 平均优势: {avg_adv:.2f}%\n")

    lines.append("## 各配置明细\n")
    for k, row in pvt.sort_values("Advantage_%", ascending=False).iterrows():
        pf, cma, adv = row["pi_flight"], row["cma_es"], row["Advantage_%"]
        lines.append(f"- {k}: π-Flight={pf:.4f}, CMA-ES={cma:.4f}, 优势={adv:+.2f}%")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[WRITE] {OUT_MD}")


if __name__ == "__main__":
    main()
