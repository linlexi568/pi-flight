from __future__ import annotations
import os
import csv
import time
import subprocess
from typing import List, Dict

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(ROOT)
PY = os.path.join(WS, ".venv", "Scripts", "python.exe")
VERIFY = os.path.join(WS, "utilities", "verify_program.py")

# 评测组合：3 个轨迹集 × 3 个扰动等级
TRAJ_PRESETS = [
    ("train_core", 20),        # 训练核心/常规
    ("test_challenge", 20),    # 挑战/泛化
    ("test_extreme", 25),      # 极端测试
]
DISTURBANCES = [None, "mild_wind", "stress"]
AGG = "harmonic"

# 待评测的方法：π-Flight（MCTS+DSL）与 CMA-ES 基线
METHODS = [
    {
        "name": "pi_flight",
        "program": os.path.join(WS, "01_pi_flight", "results", "best_program.json"),
        "compose_by_gain": True,
        "clip_D": 1.2,
    },
    {
        "name": "cma_es",
        "program": os.path.join(WS, "03_CMA-ES", "results", "best_program.json"),
        "compose_by_gain": False,
        "clip_D": 1.2,
    },
]

OUT_DIR = os.path.join(WS, "results", "summaries")
OUT_CSV = os.path.join(OUT_DIR, "comprehensive_comparison_results_collected.csv")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def run_one(program: str, traj_preset: str, disturbance: str | None, duration: int,
            compose_by_gain: bool, clip_D: float | None) -> float:
    if not os.path.isfile(program):
        raise FileNotFoundError(f"Program JSON not found: {program}")
    cmd: List[str] = [
        PY, VERIFY,
        "--program", program,
        "--traj_preset", traj_preset,
        "--aggregate", AGG,
        "--duration", str(duration),
        "--log-skip", "2",
        "--reward_profile", "pilight_boost",
    ]
    if compose_by_gain:
        cmd.append("--compose-by-gain")
    if clip_D is not None:
        cmd.extend(["--clip-D", str(clip_D)])
    if disturbance is not None:
        cmd.extend(["--disturbance", disturbance])

    print("\n[RUN]", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WS, capture_output=True, text=True)
    t1 = time.time()
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"verify_program failed with code {proc.returncode}")

    # 解析标准输出中的聚合得分行：[Verified] 聚合得分: X.XXXXXX
    score = None
    for line in proc.stdout.splitlines():
        if "[Verified]" in line and "聚合得分" in line:
            try:
                score = float(line.split(":")[-1].strip())
            except Exception:
                pass
    if score is None:
        print(proc.stdout)
        raise ValueError("Failed to parse verified score from output")

    print(f"[OK] {traj_preset} | {disturbance or 'none'} | score={score:.6f} | {t1-t0:.1f}s")
    return score


def main():
    ensure_dirs()

    rows: List[Dict[str, str]] = []
    for traj_preset, duration in TRAJ_PRESETS:
        for disturbance in DISTURBANCES:
            for m in METHODS:
                try:
                    score = run_one(
                        program=m["program"],
                        traj_preset=traj_preset,
                        disturbance=disturbance,
                        duration=duration,
                        compose_by_gain=m["compose_by_gain"],
                        clip_D=m["clip_D"],
                    )
                except Exception as e:
                    print(f"[ERR] {m['name']} {traj_preset} {disturbance}: {e}")
                    score = float("nan")

                rows.append({
                    "Method": m["name"],
                    "Trajectory_Set": traj_preset,
                    "Disturbance": disturbance or "none",
                    "Duration": str(duration),
                    "Score": f"{score:.6f}" if score == score else "NaN",
                })

    header = ["Method", "Trajectory_Set", "Disturbance", "Duration", "Score"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[WRITE] {OUT_CSV}")


if __name__ == "__main__":
    main()
