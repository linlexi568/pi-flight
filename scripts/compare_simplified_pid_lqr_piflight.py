#!/usr/bin/env python3
"""在统一的简化动力学模型上比较 PID / LQR / π-Flight。

- 动力学与 reward 计算直接复用 scripts/baselines/tune_pid_lqr.py 里的 simulate_episode
- 奖励：Safe-Control-Gym 论文中定义的二次 tracking cost (SCG_Q_DIAG / SCG_R_DIAG)
- 输出：state_cost, action_cost, true_reward (= -state_cost - action_cost), RMSE
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
for candidate in (CURRENT_DIR, ROOT, ROOT / "scripts", ROOT / "01_pi_flight"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baselines.tune_pid_lqr import (  # type: ignore
    PIDParams,
    LQRParams,
    TunablePIDController,
    TunableLQRController,
    simulate_episode,
)
from core.serialization import load_program_json  # type: ignore
from core.dsl import BinaryOpNode, TerminalNode  # type: ignore
from utils.gpu_program_executor import GPUProgramExecutor  # type: ignore

try:  # pragma: no cover
    from utils.segmented_controller import PiLightSegmentedPIDController  # type: ignore
except Exception:  # pragma: no cover
    PiLightSegmentedPIDController = None  # type: ignore


@dataclass
class EvalConfig:
    task: str = "figure8"
    duration: float = 5.0
    ctrl_freq: float = 48.0
    episodes: int = 1


class PiFlightSimplifiedController:
    """把 π-Flight DSL 程序接到简化动力学上用的控制器包装。"""

    def __init__(self, program_json: Path, control_dt: float, device: str = "cpu") -> None:
        if not program_json.is_file():
            raise FileNotFoundError(f"Pi-Flight 程序不存在: {program_json}")
        self._program = load_program_json(str(program_json))
        self._use_direct_outputs = program_uses_direct_outputs(self._program)
        self._device = torch.device(device)
        self._control_dt = float(control_dt)
        self._executor: Optional[GPUProgramExecutor] = None
        self._token: Optional[int] = None
        self._integral_states: List[Dict[str, float]] = []
        self._segmented: Optional[Any] = None

        if self._use_direct_outputs:
            self._executor = GPUProgramExecutor(device=device)
            self._prepare_executor()
            self._integral_states = [
                {
                    "err_i_x": 0.0,
                    "err_i_y": 0.0,
                    "err_i_z": 0.0,
                    "err_i_roll": 0.0,
                    "err_i_pitch": 0.0,
                    "err_i_yaw": 0.0,
                }
            ]
        else:
            self._segmented = self._build_segmented_controller()

        self._last_safe = np.zeros(4, dtype=np.float64)
        self._mad_min_fz = 0.0
        self._mad_max_fz = 7.5
        self._mad_max_xy = 0.12
        self._mad_max_yaw = 0.04
        self._mad_max_delta_fz = 1.5
        self._mad_max_delta_xy = 0.03
        self._mad_max_delta_yaw = 0.02

    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        omega: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray = None,
    ) -> np.ndarray:
        if target_vel is None:
            target_vel = np.zeros(3)

        if self._use_direct_outputs and self._executor is not None and self._token is not None:
            pos_t = torch.tensor(pos.reshape(1, 3), dtype=torch.float32, device=self._device)
            vel_t = torch.tensor(vel.reshape(1, 3), dtype=torch.float32, device=self._device)
            omega_t = torch.tensor(omega.reshape(1, 3), dtype=torch.float32, device=self._device)
            quat_np = rpy_to_quat(rpy)
            quat_t = torch.tensor(quat_np.reshape(1, 4), dtype=torch.float32, device=self._device)
            target_t = torch.tensor(target_pos.reshape(1, 3), dtype=torch.float32, device=self._device)
            use_mask = torch.ones(1, dtype=torch.bool, device=self._device)
            outputs, pos_err_t, rpy_t = self._executor.evaluate_from_raw_obs(
                self._token,
                pos_t,
                vel_t,
                omega_t,
                quat_t,
                target_t,
                self._integral_states,
                use_mask,
            )
            self._update_integral_states(pos_err_t, rpy_t)
            action = outputs[0].detach().cpu().numpy()
            return self._apply_mad(action.astype(np.float64))

        if self._segmented is None:
            self._segmented = self._build_segmented_controller()
        quat = rpy_to_quat(rpy)
        rpm, _pos_e, _rpy_e = self._segmented.computeControl(
            self._control_dt,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=omega,
            target_pos=target_pos,
        )
        forces = np.asarray(self._rpm_to_forces_local(np.asarray(rpm, dtype=np.float64)), dtype=np.float64)
        return self._apply_mad(forces)

    def close(self) -> None:
        try:
            if self._token is not None and self._executor is not None:
                self._executor.release_batch(self._token)
        except Exception:
            pass
        self._token = None
        self._segmented = None

    def reset_episode(self) -> None:
        self._last_safe.fill(0.0)
        if self._use_direct_outputs:
            if self._executor is not None and self._token is not None:
                self._executor.release_batch(self._token)
            self._prepare_executor()
            for buf in self._integral_states:
                for key in buf:
                    buf[key] = 0.0
        else:
            self._segmented = self._build_segmented_controller()

    def _apply_mad(self, raw: np.ndarray) -> np.ndarray:
        current = raw.copy()
        current[0] = np.clip(current[0], self._mad_min_fz, self._mad_max_fz)
        lateral = current[1:3]
        lat_norm = np.linalg.norm(lateral)
        if lat_norm > self._mad_max_xy and lat_norm > 1e-6:
            current[1:3] = lateral * (self._mad_max_xy / lat_norm)
        current[3] = np.clip(current[3], -self._mad_max_yaw, self._mad_max_yaw)

        delta = current - self._last_safe
        delta[0] = np.clip(delta[0], -self._mad_max_delta_fz, self._mad_max_delta_fz)
        delta[1] = np.clip(delta[1], -self._mad_max_delta_xy, self._mad_max_delta_xy)
        delta[2] = np.clip(delta[2], -self._mad_max_delta_xy, self._mad_max_delta_xy)
        delta[3] = np.clip(delta[3], -self._mad_max_delta_yaw, self._mad_max_delta_yaw)
        safe = self._last_safe + delta
        self._last_safe = safe
        return safe

    def _update_integral_states(self, pos_err_t: torch.Tensor, rpy_t: torch.Tensor) -> None:
        if not self._integral_states:
            return
        dt = self._control_dt
        buf = self._integral_states[0]
        buf["err_i_x"] += float(pos_err_t[0, 0].item()) * dt
        buf["err_i_y"] += float(pos_err_t[0, 1].item()) * dt
        buf["err_i_z"] += float(pos_err_t[0, 2].item()) * dt
        buf["err_i_roll"] += float(rpy_t[0, 0].item()) * dt
        buf["err_i_pitch"] += float(rpy_t[0, 1].item()) * dt
        buf["err_i_yaw"] += float(rpy_t[0, 2].item()) * dt

    def _prepare_executor(self) -> None:
        if self._executor is None:
            return
        if self._token is not None:
            try:
                self._executor.release_batch(self._token)
            except Exception:
                pass
        self._token = self._executor.prepare_batch([self._program])

    def _build_segmented_controller(self) -> Any:
        if PiLightSegmentedPIDController is None:
            raise RuntimeError(
                "PiLightSegmentedPIDController 未找到（缺少 utils.local_pid 依赖），"
                "无法评估仅包含增益段的程序。"
            )
        return PiLightSegmentedPIDController(
            program=self._program,
            suppress_init_print=True,
            semantics="compose_by_gain",
            min_hold_steps=2,
        )

    @staticmethod
    def _rpm_to_forces_local(rpm: np.ndarray) -> Tuple[float, float, float, float]:
        KF = 2.8e-08
        KM = 1.1e-10
        L = 0.046
        omega = np.asarray(rpm, dtype=np.float64) * (2.0 * np.pi / 60.0)
        thrust = KF * (omega ** 2)
        fz = float(np.sum(thrust))
        tx = float(L * (thrust[1] - thrust[3]))
        ty = float(L * (thrust[2] - thrust[0]))
        tz = float(KM * (omega[0] ** 2 - omega[1] ** 2 + omega[2] ** 2 - omega[3] ** 2))
        return fz, tx, ty, tz


def program_uses_direct_outputs(program: List[Dict[str, Any]]) -> bool:
    try:
        for rule in program or []:
            for action in rule.get("action", []) or []:
                if isinstance(action, BinaryOpNode) and action.op == "set":
                    left = getattr(action, "left", None)
                    if isinstance(left, TerminalNode):
                        key = str(getattr(left, "value", ""))
                        if key in {"u_fz", "u_tx", "u_ty", "u_tz"}:
                            return True
    except Exception:
        return False
    return False


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def evaluate_controller(controller, cfg: EvalConfig) -> Dict[str, float]:
    """循环运行 simulate_episode 并取平均（虽然当前动力学是确定性的）。"""
    stats: List[Dict[str, float]] = []
    for _ in range(cfg.episodes):
        if hasattr(controller, "reset_episode"):
            try:
                controller.reset_episode()  # type: ignore[attr-defined]
            except Exception:
                pass
        stats.append(simulate_episode(controller, task=cfg.task, duration=cfg.duration, ctrl_freq=cfg.ctrl_freq))

    agg = {}
    for key in stats[0].keys():
        agg[key] = float(np.mean([s[key] for s in stats]))
    return agg


def load_params_from_file(json_path: Path) -> Dict[str, Dict[str, float]]:
    if not json_path.is_file():
        raise FileNotFoundError(f"未找到参数文件: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "pid" not in data or "lqr" not in data:
        raise ValueError("参数文件里需要包含 pid 和 lqr 节点")
    return {
        "pid": data["pid"].get("params", {}),
        "lqr": data["lqr"].get("params", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统一在简化动力学上比较 PID / LQR / π-Flight")
    parser.add_argument("--program", type=str, default="results/scg_aligned/figure8_safe_control_tracking_best.json")
    parser.add_argument("--params-json", type=str, default="results/tuned_baselines_final.json")
    parser.add_argument("--task", type=str, default="figure8")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--ctrl-freq", type=float, default=48.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="results/simplified_compare_pid_lqr_piflight.json")
    args = parser.parse_args()

    cfg = EvalConfig(task=args.task, duration=args.duration, ctrl_freq=args.ctrl_freq, episodes=args.episodes)
    program_path = Path(args.program)

    params = load_params_from_file(Path(args.params_json))
    pid_params = PIDParams(**params["pid"])
    lqr_params = LQRParams(**params["lqr"])

    print("================= 简化模型对比 =================")
    print(f"task={cfg.task}, duration={cfg.duration}s, ctrl_freq={cfg.ctrl_freq}Hz, episodes={cfg.episodes}\n")

    pid_controller = TunablePIDController(pid_params)
    lqr_controller = TunableLQRController(lqr_params)
    piflight_controller = PiFlightSimplifiedController(program_path, control_dt=1.0 / cfg.ctrl_freq, device=args.device)

    try:
        pid_stats = evaluate_controller(pid_controller, cfg)
        lqr_stats = evaluate_controller(lqr_controller, cfg)
        piflight_stats = evaluate_controller(piflight_controller, cfg)
    finally:
        piflight_controller.close()

    def fmt_row(name: str, stats: Dict[str, float]) -> str:
        return (
            f"{name:<10} "
            f"{stats['true_reward']:<12.4f} "
            f"{stats['state_cost']:<12.4f} "
            f"{stats['action_cost']:<12.6f} "
            f"{stats['rmse']:<10.4f}"
        )

    print(f"{'Controller':<10} {'true_reward':<12} {'state_cost':<12} {'action_cost':<12} {'RMSE':<10}")
    print("-" * 65)
    print(fmt_row("PID", pid_stats))
    print(fmt_row("LQR", lqr_stats))
    print(fmt_row("PiFlight", piflight_stats))

    output = {
        "task": cfg.task,
        "duration": cfg.duration,
        "ctrl_freq": cfg.ctrl_freq,
        "episodes": cfg.episodes,
        "program": str(program_path),
        "pid": pid_stats,
        "lqr": lqr_stats,
        "piflight": piflight_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
