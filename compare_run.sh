#!/usr/bin/env bash
# ==============================================================================
# compare_run.sh - 统一触发 PID / PPO / Program 三种方案的评测命令
# ==============================================================================
# 使用方法：编辑下方“可编辑参数”区域（务必不要在命令行附加参数），然后运行
#   ./compare_run.sh
# 日志输出会写入 compare/logs/，运行摘要追加到 compare/run_history.csv。
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

COMPARE_DIR="$REPO_ROOT/compare"
LOG_DIR="$COMPARE_DIR/logs"
RUN_HISTORY="$COMPARE_DIR/run_history.csv"

mkdir -p "$LOG_DIR"

# ==============================
# ⚙️ 可编辑参数（禁止 CLI 传参）
# ==============================
# 将对应开关设为 1 即会执行，0 表示跳过。默认全部关闭，避免误触长时间任务。
RUN_PID=0
RUN_PPO=0
RUN_PROGRAM=0

# PID 基线命令：使用标准 PID（无程序增强）在测试集上评估
PID_LABEL="pid_baseline"
PID_COMMAND=(
  "$PYTHON_BIN" "$COMPARE_DIR/pid_baseline.py"
    --traj_preset test_challenge
    --aggregate harmonic
    --duration 20
    --reward_profile control_law_discovery
    --output "$COMPARE_DIR/logs/pid_result.json"
)

# PPO 基线命令：假设已有训练好的模型，这里只做评估
# 如需训练，请先手动运行: python 02_PPO/baseline_ppo.py --mode train --task circle
PPO_LABEL="ppo_baseline"
PPO_COMMAND=(
  "$PYTHON_BIN" "$REPO_ROOT/02_PPO/baseline_ppo.py"
    --mode eval
    --task circle
    --model-path "$REPO_ROOT/02_PPO/ppo_baseline/circle/final_model"
    --n-eval 50
)

# π-Flight 程序合成评估命令。默认复用长跑产生的 best program，可根据需要调整。
PROGRAM_LABEL="piflight_program"
PROGRAM_COMMAND=(
  "$PYTHON_BIN" "$REPO_ROOT/utilities/verify_program.py"
    --program "$REPO_ROOT/01_pi_flight/results/longrun_1000iters_20251114_001449.json"
    --traj_preset test_challenge
    --aggregate harmonic
    --disturbance mild_wind
    --duration 20
    --log-skip 2
    --compose-by-gain
    --clip-D 1.2
)

# ==============================
# 辅助函数
# ==============================
ensure_compare_dir() {
  if [[ ! -d "$COMPARE_DIR" ]]; then
    echo "[compare] 缺少 compare/ 目录，请先创建（或运行 git checkout）" >&2
    exit 1
  fi
}

append_history() {
  local label="$1"; shift
  local duration="$1"; shift
  local log_path="$1"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  if [[ ! -f "$RUN_HISTORY" ]]; then
    echo "timestamp,label,duration_sec,log_path" > "$RUN_HISTORY"
  fi
  echo "$timestamp,$label,$duration,$log_path" >> "$RUN_HISTORY"
}

run_block() {
  local label="$1"; shift
  local -a cmd=("$@")
  local stamp
  stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  local log_file="$LOG_DIR/${label}_${stamp}.log"
  echo "[compare] => $label"
  echo "[compare] 命令: ${cmd[*]}"
  local start_ts
  start_ts=$(date +%s)
  {
    echo "[compare] start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    "${cmd[@]}"
    echo "[compare] end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  } | tee "$log_file"
  local end_ts
  end_ts=$(date +%s)
  local duration=$(( end_ts - start_ts ))
  echo "[compare] $label 用时 ${duration}s"
  append_history "$label" "$duration" "$log_file"
}

main() {
  ensure_compare_dir
  [[ "$RUN_PID" == "1" ]] && run_block "$PID_LABEL" "${PID_COMMAND[@]}"
  [[ "$RUN_PPO" == "1" ]] && run_block "$PPO_LABEL" "${PPO_COMMAND[@]}"
  [[ "$RUN_PROGRAM" == "1" ]] && run_block "$PROGRAM_LABEL" "${PROGRAM_COMMAND[@]}"
  if [[ "$RUN_PID" != "1" && "$RUN_PPO" != "1" && "$RUN_PROGRAM" != "1" ]]; then
    echo "[compare] 所有任务均被关闭，请在 compare_run.sh 顶部打开至少一个 RUN_* 开关。"
  else
    echo "[compare] 全部启用任务完成，详情见 $RUN_HISTORY"
  fi
}

main
