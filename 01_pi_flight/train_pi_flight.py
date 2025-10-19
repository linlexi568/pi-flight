"""Standalone training/search entry for PI-Flight (MCTS over DSL rules).

迁移自 01_pi_light/train_pi_light.py，并改为默认保存到 01_pi_flight/results。
"""
from __future__ import annotations
import argparse, os, time, json, io, contextlib, warnings, sys, pathlib
import numpy as np

# --- Early deep quiet detection (before gym_pybullet_drones imports) ---
_DEEP_QUIET_EARLY = ('--deep-quiet' in sys.argv)
_saved_out = _saved_err = _devnull = None  # sentinel for analyzer
if _DEEP_QUIET_EARLY:
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    warnings.filterwarnings("ignore", module="pkg_resources")
    try:
        import os as _osq
        _devnull = open(_osq.devnull, 'w')
        _saved_out = _osq.dup(1)
        _saved_err = _osq.dup(2)
        _osq.dup2(_devnull.fileno(), 1)
        _osq.dup2(_devnull.fileno(), 2)
    except Exception:
        _DEEP_QUIET_EARLY = False

from typing import List, Dict, Any, Tuple

# Restore deprecated numpy aliases required by older dependencies like cma
if not hasattr(np, 'Inf'):
    np.Inf = np.inf  # type: ignore[attr-defined]
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf  # type: ignore[attr-defined]
from collections import OrderedDict
# Limit BLAS threads early to reduce per-process memory/threads pressure
try:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')
except Exception:
    pass
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Restore FDs after imports if we silenced them
if _DEEP_QUIET_EARLY:
    try:
        import os as _osr
        if _saved_out is not None:
            _osr.dup2(_saved_out, 1); _osr.close(_saved_out)
        if _saved_err is not None:
            _osr.dup2(_saved_err, 2); _osr.close(_saved_err)
        if _devnull is not None:
            _devnull.close()
    except Exception:
        pass

# 兼容两种用法：包内相对导入 或 直接脚本执行
try:
    if __package__:
        from . import MCTS_Agent, PiLightSegmentedPIDController, BinaryOpNode, UnaryOpNode, TerminalNode
        from .serialization import save_program_json, save_search_history, deserialize_program
    else:
        raise ImportError
except Exception:
    import importlib.util as _ilu, importlib as _il, pathlib as _pl, sys as _sys
    _CUR = _pl.Path(__file__).resolve()
    _PKG_DIR = _CUR.parent
    _PKG_NAME = 'pi_flight_local'
    _spec = _ilu.spec_from_file_location(_PKG_NAME, str(_PKG_DIR / '__init__.py'), submodule_search_locations=[str(_PKG_DIR)])
    if _spec is None or _spec.loader is None:
        raise ImportError('Failed to locate 01_pi_flight package files')
    _mod = _ilu.module_from_spec(_spec)  # type: ignore
    _sys.modules[_PKG_NAME] = _mod
    _spec.loader.exec_module(_mod)       # type: ignore
    MCTS_Agent = getattr(_mod, 'MCTS_Agent')
    PiLightSegmentedPIDController = getattr(_mod, 'PiLightSegmentedPIDController')
    BinaryOpNode = getattr(_mod, 'BinaryOpNode')
    UnaryOpNode = getattr(_mod, 'UnaryOpNode')
    TerminalNode = getattr(_mod, 'TerminalNode')
    _ser_mod = _il.import_module(f'{_PKG_NAME}.serialization')
    save_program_json = getattr(_ser_mod, 'save_program_json')
    save_search_history = getattr(_ser_mod, 'save_search_history')
    deserialize_program = getattr(_ser_mod, 'deserialize_program')

# 确保优先导入工作区根目录下的 test.py（避免与 Python 标准库 test 包冲突）
try:
    import importlib, sys as _sys, pathlib as _pl
    _CURR = _pl.Path(__file__).resolve(); _ROOTP = _CURR.parent.parent
    if str(_ROOTP) not in _sys.path:
        _sys.path.insert(0, str(_ROOTP))
    SimulationTester = importlib.import_module('test').SimulationTester  # type: ignore
except Exception as _te:
    raise ImportError(f"Failed to import local SimulationTester from test.py: {_te}")

from utilities.reward_profiles import get_reward_profile, describe_profile
# TrustRegionManager import compatible with both package and script modes
# LAZY IMPORT: defer to avoid circular import / module initialization deadlock
TrustRegionManager = None  # type: ignore
def _get_trust_region_manager():
    global TrustRegionManager
    if TrustRegionManager is not None:
        return TrustRegionManager
    try:
        if __package__:
            from .trust_region import TrustRegionManager as TRM  # type: ignore
        else:
            raise ImportError
    except Exception:
        try:
            import importlib as _il_tr
            TRM = getattr(_il_tr.import_module(f"{_PKG_NAME}.trust_region"), 'TrustRegionManager')  # type: ignore
        except Exception:
            TRM = None  # type: ignore
    TrustRegionManager = TRM
    return TrustRegionManager

def build_trajectory(name: str):
    if name == 'figure8':
        return { 'type': 'figure_8','initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8,'B': 0.5,'period': 12}}
    elif name == 'helix':
        return { 'type': 'helix','initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7,'period': 10,'v_z': 0.15}}
    elif name == 'circle':
        return { 'type': 'circle','initial_xyz': [0, 0, 0.8], 'params': {'R': 0.9,'period': 10}}
    elif name == 'square':
        return { 'type': 'square','initial_xyz': [0, 0, 0.8], 'params': {'side_len': 1.2,'period': 12,'corner_hold': 0.5}}
    elif name == 'step_hover':
        return { 'type': 'step_hover','initial_xyz': [0, 0, 0.6], 'params': {'z2': 1.2,'switch_time': 6.0}}
    elif name == 'spiral_out':
        return { 'type': 'spiral_out','initial_xyz': [0, 0, 0.6], 'params': {'R0': 0.2,'k': 0.05,'period': 9,'v_z':0.02}}
    elif name == 'zigzag3d':
        return { 'type': 'zigzag3d','initial_xyz': [0, 0, 0.7], 'params': {'amplitude': 0.8,'segments': 6,'z_inc': 0.08,'period': 14.0}}
    elif name == 'lemniscate3d':
        return { 'type': 'lemniscate3d','initial_xyz': [0, 0, 0.7], 'params': {'a': 0.9,'period': 16.0,'z_amp': 0.25}}
    elif name in ('random_wp','random_waypoints'):
        return { 'type': 'random_waypoints','initial_xyz': [0, 0, 0.8], 'params': {'hold_time': 1.2, 'transition': 'linear'}}
    elif name == 'spiral_in_out':
        return { 'type': 'spiral_in_out','initial_xyz': [0, 0, 0.7], 'params': {'R_in': 0.9,'R_out': 0.2,'period': 14.0,'z_wave': 0.15}}
    elif name == 'stairs':
        return { 'type': 'stairs','initial_xyz': [0, 0, 0.6], 'params': {'levels': [0.6, 0.9, 1.2], 'segment_time': 3.0}}
    elif name == 'coupled_surface':
        return { 'type': 'coupled_surface','initial_xyz': [0, 0, 0.8], 'params': {'ax': 0.9,'ay': 0.7,'f1': 1.0,'f2': 2.0,'phase': 1.0472,'z_base': 0.8,'z_amp': 0.25,'surf_amp': 0.15}}
    # 测试集极端版本 (训练/测试分离)
    elif name == 'coupled_surface_extreme':
        return { 'type': 'coupled_surface','initial_xyz': [0, 0, 0.9], 'params': {'ax': 1.1,'ay': 0.9,'f1': 1.5,'f2': 3.0,'phase': 0.7,'z_base': 0.9,'z_amp': 0.35,'surf_amp': 0.22}}
    elif name == 'zigzag3d_aggressive':
        return { 'type': 'zigzag3d','initial_xyz': [0, 0, 0.6], 'params': {'amplitude': 1.1,'segments': 8,'z_inc': 0.12,'period': 10.0}}
    elif name == 'lemniscate3d_wild':
        return { 'type': 'lemniscate3d','initial_xyz': [0, 0, 0.6], 'params': {'a': 1.2,'period': 12.0,'z_amp': 0.40}}
    elif name == 'spiral_chaotic':
        return { 'type': 'spiral_in_out','initial_xyz': [0, 0, 0.65], 'params': {'R_in': 1.1,'R_out': 0.15,'period': 10.0,'z_wave': 0.25}}
    elif name == 'stairs_harsh':
        return { 'type': 'stairs','initial_xyz': [0, 0, 0.5], 'params': {'levels': [0.5, 0.8, 1.1, 1.4],'segment_time': 2.2}}
    else:
        raise ValueError(f"Unknown trajectory: {name}")

def build_disturbances(preset: str | None):
    if not preset:
        return []
    if preset == 'mild_wind':
        return [
            {'type': 'SUSTAINED_WIND','info':'mild','start_time':3.0,'end_time':6.0,'force':[0.01,0,0]},
            {'type': 'PULSE','time':8.0,'force':[0.02,-0.01,0],'info':'pulse'}
        ]
    if preset == 'stress':
        return [
            {'type': 'SUSTAINED_WIND','info':'stress:steady_wind','start_time':2.0,'end_time':6.0,'force':[0.015,0.0,0]},
            {'type': 'GUSTY_WIND','info':'stress:gusty_wind','start_time':7.0,'end_time':11.0,'base_force':[0,-0.01,0],'gust_frequency':9.0,'gust_amplitude':0.012},
            {'type': 'MASS_CHANGE','info':'stress:mass_up','time':12.0,'mass_multiplier':1.15},
            {'type': 'PULSE','info':'stress:pulse','time':14.0,'force':[-0.02,0.02,0]}
        ]
    raise ValueError(f"Unknown disturbance preset: {preset}")

def build_preset_list(preset: str) -> list[str]:
    """Return trajectory name list for a given preset (used by standardized test verification)."""
    if preset == 'train_core':
        return ['figure8','helix','circle','square','step_hover','spiral_out']
    if preset in ('test_challenge',):
        return ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface']
    if preset == 'test_extreme':
        return ['coupled_surface_extreme','zigzag3d_aggressive','lemniscate3d_wild','spiral_chaotic','stairs_harsh']
    if preset == 'full_eval':
        return ['figure8','helix','circle','square','step_hover','spiral_out',
                'zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface']
    if preset == 'pi_strong_train':
        return ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface']
    raise ValueError(f"Unknown preset: {preset}")

def parse_args():
    p = argparse.ArgumentParser(description='Train/Search PI-Flight rules using MCTS')
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--traj', type=str, default='figure8',
                   choices=['figure8','helix','circle','square','step_hover','spiral_out',
                            'zigzag3d','lemniscate3d','random_wp','random_waypoints','spiral_in_out','stairs','coupled_surface'])
    p.add_argument('--traj_list', type=str, nargs='*', default=None)
    p.add_argument('--traj_preset', type=str, default=None,
                   choices=['pi_strong_train'])
    p.add_argument('--aggregate', type=str, default='harmonic', choices=['mean','min','harmonic'])
    p.add_argument('--duration', type=int, default=20)
    p.add_argument('--disturbance', type=str, default='mild_wind', choices=[None,'mild_wind','stress'])
    p.add_argument('--save-every', type=int, default=0)
    p.add_argument('--save-program', type=str, default='01_pi_flight/results/best_program.json')
    p.add_argument('--save-history', type=str, default='01_pi_flight/results/search_history.json')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--report', type=int, default=20)
    # MCTS config
    p.add_argument('--mcts_max_depth', type=int, default=20)
    p.add_argument('--rollout_depth', type=int, default=4)
    # 已默认关闭复杂度惩罚
    p.add_argument('--complexity_penalty', type=float, default=0.0)
    p.add_argument('--pw_alpha', type=float, default=0.6)
    p.add_argument('--pw_c', type=float, default=1.5)
    # 复杂度调度（默认关闭）
    p.add_argument('--complexity-min-scale', type=float, default=0.0)
    p.add_argument('--complexity-max-scale', type=float, default=0.0)
    p.add_argument('--complexity-ramp-start', type=float, default=0.0)
    p.add_argument('--complexity-ramp-end', type=float, default=0.0)
    p.add_argument('--no-complexity-penalty', action='store_true')
    p.add_argument('--reward_profile', type=str, default='pilight_boost', choices=['default','pilight_boost'])
    p.add_argument('--tqdm', action='store_true')
    p.add_argument('--banner-every', type=int, default=0)
    p.add_argument('--quiet-eval', action='store_true')
    p.add_argument('--warm_start_cmaes', action='store_true')
    p.add_argument('--warm_start_path', type=str, default='03_CMA-ES/results/best_program.json')
    p.add_argument('--warm-start-program', type=str, default=None)
    # 默认在 warm start 后补齐到 min-rules-guard，推动早期分裂
    p.add_argument('--pad-after-warm-start', action='store_true', default=True)
    # 性能
    p.add_argument('--log-skip', type=int, default=2)
    p.add_argument('--in-memory-log', action='store_true')
    p.add_argument('--short-duration', type=int, default=6)
    p.add_argument('--full-duration', type=int, default=20)
    p.add_argument('--short-frac', type=float, default=0.4)
    # 抗过拟合：周期性全时长探针混合（默认关闭）
    p.add_argument('--fullmix-every', type=int, default=0, help='每隔N次迭代，把全时长评估按比例混入训练目标；0=禁用')
    p.add_argument('--fullmix-frac', type=float, default=0.0, help='混合比例：value = (1-frac)*short + frac*full（默认0禁用）')
    p.add_argument('--fullmix-ramp-start', type=float, default=0.0, help='按进度从此处开始线性爬升到 fullmix-frac（0-1，默认0）')
    p.add_argument('--fullmix-ramp-end', type=float, default=0.0, help='在此进度达到 fullmix-frac（0-1，默认0=恒定比例）')
    # 缓存/复用
    p.add_argument('--cache-size', type=int, default=2048)
    p.add_argument('--reuse-env', action='store_true')
    p.add_argument('--quiet-sim', action='store_true')
    p.add_argument('--deep-quiet', action='store_true')
    # 并行与批次
    p.add_argument('--parallel-traj', action='store_true')
    p.add_argument('--num-workers', type=int, default=0, help='并行评估的进程数：0=禁用进程池（单进程），>1=启用固定进程数，-1=自动(约等于CPU-1)')
    p.add_argument('--traj-batch-size', type=int, default=0)
    # 迭代简报
    p.add_argument('--iter-log-file', type=str, default=None)
    # 测试集实时验证
    p.add_argument('--test-verify-every', type=int, default=0, help='每隔N次迭代在测试集上验证；0=禁用')
    p.add_argument('--test-traj-preset', type=str, default='test_challenge', help='测试集轨迹预设')
    p.add_argument('--test-aggregate', type=str, default='harmonic', choices=['mean','min','harmonic'])
    p.add_argument('--test-disturbance', type=str, default='mild_wind', choices=[None,'mild_wind','stress'])
    p.add_argument('--test-duration', type=int, default=20, help='测试集评估时长')
    p.add_argument('--test-clip-D', type=float, default=1.2, help='测试集D裁剪')
    # CMA-ES 混合训练（MCTS负责结构，CMA负责参数微调）
    p.add_argument('--cma-refine-every', type=int, default=0, help='每隔N次MCTS迭代，对当前最优程序的PID参数做一轮CMA-ES微调；0=禁用')
    p.add_argument('--cma-popsize', type=int, default=8, help='CMA-ES种群大小（越小越快）')
    p.add_argument('--cma-maxiter', type=int, default=20, help='CMA-ES每次微调的最大迭代数')
    p.add_argument('--cma-sigma', type=float, default=0.15, help='CMA-ES初始步长（相对于当前参数）')
    p.add_argument('--cma-parallel', action='store_true', help='CMA-ES评估是否并行（实验性）')
    # 复核/门控与标准化测试验证已移除，接口精简为训练集度量
    # 分段增长/探索
    p.add_argument('--min-rules-guard', type=int, default=2)
    p.add_argument('--max-rules', type=int, default=8)
    p.add_argument('--add-rule-bias-base', type=int, default=2)
    p.add_argument('--min-rules-final', type=int, default=None)
    p.add_argument('--min-rules-ramp-start', type=float, default=0.30)
    p.add_argument('--min-rules-ramp-end', type=float, default=0.70)
    p.add_argument('--epsilon-max', type=float, default=0.25)
    p.add_argument('--epsilon-end-progress', type=float, default=0.30)
    p.add_argument('--swap-span', type=int, default=4)
    p.add_argument('--stagnation-window', type=int, default=0)
    p.add_argument('--epsilon-rebound', type=float, default=0.18)
    p.add_argument('--rebound-iters', type=int, default=80)
    p.add_argument('--rebound-decay-iters', type=int, default=0)
    p.add_argument('--rebound-target-eps', type=float, default=0.12)
    p.add_argument('--stagnation-seconds', type=int, default=0)
    p.add_argument('--epsilon-rebound-target', type=float, default=0.0)
    p.add_argument('--time-rebound-iters', type=int, default=0)
    p.add_argument('--diversity-bonus-max', type=float, default=0.0)
    p.add_argument('--diversity-end-progress', type=float, default=0.30)
    p.add_argument('--strict-bonus-scale', type=float, default=0.0)
    p.add_argument('--prefer-more-rules-tie-delta', type=float, default=0.0)
    p.add_argument('--prefer-fewer-rules-tie-delta', type=float, default=0.0)
    p.add_argument('--full-action-prob', type=float, default=0.0)
    p.add_argument('--allowed-cond-unaries', type=str, default='identity,abs')
    p.add_argument('--trig-as-phase-window', action='store_true')
    p.add_argument('--trig-lt-max', type=float, default=0.25)
    p.add_argument('--compose-by-gain', action='store_true')
    p.add_argument('--semantics', type=str, default=None, choices=[None,'first_match','compose_by_gain','blend_topk'])
    p.add_argument('--require-k', type=int, default=0)
    p.add_argument('--blend-topk-k', type=int, default=2)
    p.add_argument('--gain-slew-limit', type=str, default=None)
    p.add_argument('--min-hold-steps', type=int, default=0)
    p.add_argument('--clip-P', type=float, default=None)
    p.add_argument('--clip-I', type=float, default=None)
    p.add_argument('--clip-D', type=float, default=1.2)
    p.add_argument('--overlap-penalty', type=float, default=0.0)
    p.add_argument('--conflict-penalty', type=float, default=0.0)
    p.add_argument('--auto-unfreeze-patience', type=int, default=0)
    p.add_argument('--auto-unfreeze-steps', type=int, default=0)
    p.add_argument('--auto-unfreeze-penalty-scale', type=float, default=0.6)
    p.add_argument('--auto-unfreeze-eps-boost', type=float, default=0.15)
    # CMA-ES 联合调参、TR、先验与候补优化等扩展已移除，接口精简
    # ML-driven dynamic tuning (OFF by default)
    p.add_argument('--ml-scheduler', type=str, default='none', choices=['none','heuristic','nn'], help='启用基于ML的MCTS动态调参（默认关闭）')
    p.add_argument('--ml-interval', type=int, default=5, help='每隔N次迭代执行一次调参（默认5）')
    p.add_argument('--ml-warmup-iters', type=int, default=10, help='前N次迭代不做调参，仅收集上下文（默认10）')
    p.add_argument('--ml-path', type=str, default='01_pi_flight/results/nn_trained/ml_sched.pt', help='当 --ml-scheduler=nn 时的模型路径（TorchScript）')
    p.add_argument('--ml-strategy', type=str, default='absolute', choices=['absolute','delta'], help='更新方式：绝对覆盖或增量')
    p.add_argument('--ml-allowed', type=str, default='pw_alpha,pw_c,_puct_enable,_puct_c,_edit_prior_c,_dirichlet_eps,_value_mix_lambda,_full_action_prob,_prefer_more_rules_tie_delta,_prefer_fewer_rules_tie_delta,_add_rule_bias_base,_epsilon_max', help='允许被ML更新的参数白名单，逗号分隔')
    p.add_argument('--ml-safe-bounds', type=str, default='pw_alpha:0.4,1.0;pw_c:0.8,2.0;_puct_c:0.5,2.5;_dirichlet_eps:0.0,0.5;_edit_prior_c:0.0,1.0;_value_mix_lambda:0.0,0.3;_full_action_prob:0.0,0.9;_prefer_more_rules_tie_delta:0.0,0.1;_prefer_fewer_rules_tie_delta:0.0,0.1;_epsilon_max:0.05,0.6;_add_rule_bias_base:1,16', help='安全边界，格式 name:lo,hi;name2:lo,hi')
    p.add_argument('--ml-log', action='store_true', help='打印ML调参变更日志')
    p.add_argument('--ml-dump-csv', type=str, default='', help='将 ML 调参训练样本追加写入 CSV 路径（列固定，便于离线监督训练）')
    # Online policy training 与手动 AlphaZero-lite 旋钮已移除，统一交由 ML 层
    # ML 独占模式：隐藏手动 MCTS 旋钮，交由 ML 调度
    p.add_argument('--ml-exclusive', action='store_true', help='独占模式：忽略 MCTS 手动超参，统一由 ML 调度（AlphaZero-lite 默认）')
    args = p.parse_args()
    # 标准化默认：不指定 --semantics 则默认 compose_by_gain
    try:
        sem = getattr(args, 'semantics', None)
        if sem in (None, 'None'):
            setattr(args, 'compose_by_gain', True)
        else:
            s = str(sem).strip().lower()
            setattr(args, 'compose_by_gain', (s == 'compose_by_gain'))
    except Exception:
        pass
    return args

def _worker_evaluate_single(packed):
    (traj, program, dur, suppress, deep_quiet, disturbances, reward_weights, log_skip, in_memory, compose_by_gain, clip_P, clip_I, clip_D, pen_overlap, pen_conflict, semantics, require_k, blend_topk_k, gain_slew_limit, min_hold_steps) = packed
    cur = pathlib.Path(__file__).resolve(); root = cur.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    def _core():
        if deep_quiet:
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
            warnings.filterwarnings("ignore", module="pkg_resources")
        if deep_quiet:
            _dn=_o=_e=None
            try:
                import os as _osw
                _dn = open(_osw.devnull,'w'); _o=_osw.dup(1); _e=_osw.dup(2); _osw.dup2(_dn.fileno(),1); _osw.dup2(_dn.fileno(),2)
            except Exception:
                _dn=_o=_e=None
        from gym_pybullet_drones.utils.enums import DroneModel as _DM  # type: ignore
        # Fallback chain for controller import (prefer pi_flight)
        try:
            from pi_flight import PiLightSegmentedPIDController as _PLC  # type: ignore
        except Exception:
            try:
                from segmented_controller import PiLightSegmentedPIDController as _PLC  # type: ignore
            except Exception:
                import importlib
                _PLC = importlib.import_module('01_pi_flight.segmented_controller').PiLightSegmentedPIDController  # type: ignore
        from test import SimulationTester as _ST  # type: ignore
        if deep_quiet:
            try:
                import os as _osw2
                _o_loc = locals().get('_o'); _e_loc = locals().get('_e'); _dn_loc = locals().get('_dn')
                if _o_loc is not None: _osw2.dup2(_o_loc,1); _osw2.close(_o_loc)
                if _e_loc is not None: _osw2.dup2(_e_loc,2); _osw2.close(_e_loc)
                if _dn_loc is not None: _dn_loc.close()
            except Exception:
                pass
        controller = _PLC(
            drone_model=_DM("cf2x"),
            program=program,
            suppress_init_print=suppress or deep_quiet,
            compose_by_gain=bool(compose_by_gain),
            clip_P=clip_P,
            clip_I=clip_I,
            clip_D=clip_D,
            semantics=semantics,
            require_k=int(require_k or 0),
            blend_topk_k=int(blend_topk_k or 2),
            gain_slew_limit=gain_slew_limit,
            min_hold_steps=int(min_hold_steps or 0)
        )
        tester = _ST(
            controller=controller,
            test_scenarios=disturbances,
            output_folder='01_pi_flight/results/mcts_eval',
            gui=False,
            weights=reward_weights,
            trajectory=traj,
            duration_sec=dur,
            log_skip=log_skip,
            in_memory=in_memory,
            quiet=(suppress or deep_quiet)
        )
        reward = tester.run()
        try:
            if (pen_overlap and pen_overlap>0) or (pen_conflict and pen_conflict>0):
                metrics = controller.get_overlap_metrics()
                mean_overlap = float(metrics.get('mean_overlap', 1.0))
                mean_action_diff = float(metrics.get('mean_action_diff', 0.0))
                reward = float(reward) - float(mean_overlap) * float(pen_overlap or 0.0) - float(mean_action_diff) * float(pen_conflict or 0.0)
        except Exception:
            pass
        return reward
    if suppress or deep_quiet:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return _core()
    return _core()

def main():
    args = parse_args()
    # 迭代日志：默认写到 01_pi_flight
    from pathlib import Path as _Path
    iter_log_path = getattr(args, 'iter_log_file', None) or str(_Path('01_pi_flight')/ 'results' / 'iter_log.csv')
    _Path(iter_log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(iter_log_path, 'w', encoding='utf-8') as f:
        f.write('iter,short_best,rule_count,elapsed_s,it_per_s,epsilon,rebound_active\n')

    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed); np.random.seed(args.seed)
    if getattr(args, 'deep_quiet', False):
        warnings.filterwarnings("ignore", module="pkg_resources")
        import io as _io, contextlib as _ctx
        _silent = _io.StringIO()
        with _ctx.redirect_stdout(_silent), _ctx.redirect_stderr(_silent):
            try:
                import pybullet  # noqa: F401
            except Exception:
                pass

    DSL_VARIABLES = [
        'err_p_roll', 'err_p_pitch', 'err_d_roll', 'err_d_pitch',
        'ang_vel_x', 'ang_vel_y', 'err_i_roll', 'err_i_pitch',
        'pos_err_x', 'pos_err_y', 'pos_err_z',
        'err_i_x', 'err_i_y', 'err_i_z',
        'pos_err_xy', 'rpy_err_mag', 'ang_vel_mag', 'pos_err_z_abs'
    ]
    # 增加 PID 参数的关键区域密度 (0.5-2.5)
    DSL_CONSTANTS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 5.0]
    DSL_OPERATORS = ['+', '-', '*', 'abs', '>', '<', 'max', 'min', 'sin', 'cos', 'tan', 'log1p', 'sqrt']

    if args.traj_list:
        traj_names = args.traj_list
    else:
        if getattr(args, 'traj_preset', None) == 'pi_strong_train':
            traj_names = ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface']
        else:
            traj_names = [args.traj]
    trajectories = [build_trajectory(n) for n in traj_names]
    disturbances = build_disturbances(args.disturbance)
    reward_weights, reward_ks = get_reward_profile(args.reward_profile)
    print(describe_profile(args.reward_profile))

    # 封闭 MCTS 预设：由 ML 层自适应，无需预设

    short_iters = int(args.iters * max(0.0, min(1.0, args.short_frac)))

    def _ast_to_str_local(node):
        if isinstance(node, BinaryOpNode):
            return f"({_ast_to_str_local(node.left)} {node.op} {_ast_to_str_local(node.right)})"
        if isinstance(node, UnaryOpNode):
            return f"{node.op}({_ast_to_str_local(node.child)})"
        if isinstance(node, TerminalNode):
            return str(node.value)
        return str(node)
    def hash_program(program: list) -> str:
        import hashlib as _hl
        parts=[]
        for rule in program:
            cond=_ast_to_str_local(rule['condition'])
            acts=[]
            for a in rule['action']:
                if isinstance(a,BinaryOpNode) and a.op=='set' and isinstance(a.left,TerminalNode) and isinstance(a.right,TerminalNode):
                    acts.append(f"{a.left.value}:{a.right.value}")
            parts.append(cond+"|"+",".join(sorted(acts)))
        raw="||".join(parts)
        return _hl.sha1(raw.encode('utf-8')).hexdigest()
    class LRUCache:
        def __init__(self, capacity:int):
            self.capacity=capacity
            self.store: OrderedDict[str, float] = OrderedDict()
        def get(self, k:str):
            if k not in self.store:
                return None
            v=self.store.pop(k)
            self.store[k]=v
            return v
        def put(self, k:str, v:float):
            if self.capacity<=0:
                return
            if k in self.store:
                self.store.pop(k)
            elif len(self.store)>=self.capacity:
                self.store.popitem(last=False)
            self.store[k]=v
        def __len__(self):
            return len(self.store)
    cache = LRUCache(args.cache_size)
    true_holder: Dict[str, Any] = {'map': {}, 'version': 0}
    def _record_true(program: list, val: float):
        try:
            h = hash_program(program)
            true_holder['map'][h] = float(val)
            true_holder['version'] = int(true_holder.get('version', 0)) + 1
        except Exception:
            pass

    env_pool: Dict[Tuple[int,int], Any] = {}
    agent_holder: Dict[str, Any] = {'agent': None}
    pool_holder: Dict[str, Any] = {'pool': None, 'enabled': False, 'worker_n': 0}
    if args.parallel_traj:
        try:
            import multiprocessing as mp
            # Interpret num-workers semantics:
            #   0 -> disable pool (single-process)
            #  >1 -> exactly that many workers
            #  -1 -> auto (cpu_count-1, at least 2)
            # prefer multiprocessing.cpu_count() to avoid static analyzers complaining about os
            try:
                _cpu_n = int(mp.cpu_count())  # type: ignore[attr-defined]
            except Exception:
                _cpu_n = 2
            auto_n = max(2, (_cpu_n or 2) - 1)
            if args.num_workers == 0:
                pool_holder['worker_n'] = 0
                pool_holder['enabled'] = False
                print('[Parallel] num-workers=0 -> 禁用进程池（单进程评估）')
            elif args.num_workers < 0:
                pool_holder['worker_n'] = auto_n
                pool_holder['enabled'] = (pool_holder['worker_n'] > 1 and len(trajectories) > 1)
            else:
                pool_holder['worker_n'] = int(args.num_workers)
                pool_holder['enabled'] = (pool_holder['worker_n'] > 1 and len(trajectories) > 1)
            if pool_holder['enabled']:
                ctx = mp.get_context('spawn')
                pool_holder['pool'] = ctx.Pool(processes=pool_holder['worker_n'])
        except Exception as _pe:
            print(f"[Parallel][WARN] 创建进程池失败，回退单进程: {_pe}")
            pool_holder['enabled'] = False

    T = len(trajectories)
    traj_batch_size = args.traj_batch_size if args.traj_batch_size and args.traj_batch_size>0 else T
    traj_batch_size = min(max(1, traj_batch_size), T)
    def get_traj_batch_for_iter(iter_idx: int) -> Tuple[list, str]:
        if traj_batch_size >= T:
            return list(range(T)), f"allT{T}"
        start = (iter_idx * traj_batch_size) % T
        idxs = [(start + k) % T for k in range(traj_batch_size)]
        batch_id = "b" + "-".join(str(i) for i in idxs)
        return idxs, batch_id

    def dynamic_eval(program: list):
        ag = agent_holder['agent']
        cur_it = getattr(ag, 'total_iterations_done', 0) if ag is not None else 0
        dur = args.full_duration if cur_it >= short_iters else args.short_duration
        if dur == args.short_duration:
            idxs = list(range(T))
            batch_id = f"allT{T}_short"
            selected_trajs = [trajectories[i] for i in idxs]
        else:
            idxs, batch_id = get_traj_batch_for_iter(cur_it)
            selected_trajs = [trajectories[i] for i in idxs]
        if ag is not None and hasattr(ag, '_tt_salt'):
            try:
                ag._tt_salt = f"{dur}|{batch_id}"
            except Exception:
                pass
        prog_hash = hash_program(program)
        tv = int(true_holder.get('version', 0))
        ver_suffix = f"|TV{tv}" if float(getattr(args, 'true_mixin_alpha', 0.0) or 0.0) > 0.0 else ""
        cache_key = f"{dur}|{batch_id}|{prog_hash}{ver_suffix}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        scores = []
        if pool_holder['enabled']:
            packs = [(
                traj, program, dur, args.quiet_sim, args.deep_quiet,
                disturbances, reward_weights, args.log_skip, args.in_memory_log,
                bool(getattr(args, 'compose_by_gain', False)),
                getattr(args,'clip_P',None), getattr(args,'clip_I',None), getattr(args,'clip_D',None),
                float(getattr(args,'overlap_penalty',0.0) or 0.0), float(getattr(args,'conflict_penalty',0.0) or 0.0),
                getattr(args,'semantics', None), int(getattr(args,'require_k',0) or 0), int(getattr(args,'blend_topk_k',2) or 2),
                getattr(args,'gain_slew_limit', None), int(getattr(args,'min_hold_steps',0) or 0)
            ) for traj in selected_trajs]
            try:
                scores = pool_holder['pool'].map(_worker_evaluate_single, packs)  # type: ignore
            except Exception as _pm:
                print(f"[Parallel][WARN] 进程池执行失败，回退单进程: {_pm}")
                pool_holder['enabled'] = False
                for traj in selected_trajs:
                    scores.append(_worker_evaluate_single((
                        traj, program, dur, args.quiet_sim, args.deep_quiet,
                        disturbances, reward_weights, args.log_skip, args.in_memory_log,
                        bool(getattr(args, 'compose_by_gain', False)),
                        getattr(args,'clip_P',None), getattr(args,'clip_I',None), getattr(args,'clip_D',None),
                        float(getattr(args,'overlap_penalty',0.0) or 0.0), float(getattr(args,'conflict_penalty',0.0) or 0.0),
                        getattr(args,'semantics', None), int(getattr(args,'require_k',0) or 0), int(getattr(args,'blend_topk_k',2) or 2),
                        getattr(args,'gain_slew_limit', None), int(getattr(args,'min_hold_steps',0) or 0)
                    )))
        else:
            for bi, ti in enumerate(idxs):
                traj = trajectories[ti]
                controller = PiLightSegmentedPIDController(
                    drone_model=DroneModel("cf2x"), program=program,
                    suppress_init_print=args.quiet_sim,
                    compose_by_gain=bool(getattr(args, 'compose_by_gain', False)),
                    clip_P=getattr(args,'clip_P',None), clip_I=getattr(args,'clip_I',None), clip_D=getattr(args,'clip_D',None),
                    semantics=getattr(args,'semantics', None), require_k=int(getattr(args,'require_k',0) or 0), blend_topk_k=int(getattr(args,'blend_topk_k',2) or 2),
                    gain_slew_limit=getattr(args,'gain_slew_limit', None), min_hold_steps=int(getattr(args,'min_hold_steps',0) or 0)
                )
                external_env = None
                if args.reuse_env:
                    key = (dur, ti)
                    if key not in env_pool:
                        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
                        from gym_pybullet_drones.envs.BaseAviary import Physics
                        import numpy as _np
                        init_xyz = _np.array([traj.get('initial_xyz', [0,0,0.5])])
                        env_pool[key] = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=init_xyz,
                                                   physics=Physics("pyb"), pyb_freq=240, ctrl_freq=48, gui=False, record=False)
                    external_env = env_pool[key]
                tester = SimulationTester(controller=controller,
                                          test_scenarios=disturbances,
                                          output_folder='01_pi_flight/results/mcts_eval',
                                          gui=False,
                                          weights=reward_weights,
                                          trajectory=traj,
                                          duration_sec=dur,
                                          log_skip=args.log_skip,
                                          in_memory=args.in_memory_log,
                                          external_env=external_env,
                                          reuse_reset=args.reuse_env,
                                          quiet=args.quiet_sim)
                rew = tester.run()
                try:
                    if (getattr(args,'overlap_penalty',0.0) or 0.0)>0 or (getattr(args,'conflict_penalty',0.0) or 0.0)>0:
                        metrics = controller.get_overlap_metrics()
                        mean_overlap = float(metrics.get('mean_overlap', 1.0))
                        mean_action_diff = float(metrics.get('mean_action_diff', 0.0))
                        rew = float(rew) - float(mean_overlap) * float(getattr(args,'overlap_penalty',0.0) or 0.0) - float(mean_action_diff) * float(getattr(args,'conflict_penalty',0.0) or 0.0)
                except Exception:
                    pass
                scores.append(rew)
        if args.aggregate == 'mean':
            value = float(sum(scores)/len(scores))
        elif args.aggregate == 'min':
            value = float(min(scores))
        elif args.aggregate == 'harmonic':
            import math
            value = len(scores)/sum(1/(s+1e-9) for s in scores)
        else:
            value = float(sum(scores)/len(scores))

        # 抗过拟合：周期性全时长探针混合
        _mix_used = False
        try:
            _mix_every = int(getattr(args, 'fullmix_every', 0) or 0)
            _mix_frac_cfg = float(getattr(args, 'fullmix_frac', 0.0) or 0.0)
            if _mix_every > 0 and _mix_frac_cfg > 0.0:
                _progress = float(cur_it / max(1, total))
                _rs = float(getattr(args, 'fullmix_ramp_start', 0.0) or 0.0)
                _re = float(getattr(args, 'fullmix_ramp_end', 0.0) or 0.0)
                if _re > _rs and _re > 0.0:
                    _k = (_progress - _rs) / max(1e-9, (_re - _rs))
                    _k = min(1.0, max(0.0, _k))
                    _mix_frac = _mix_frac_cfg * _k
                else:
                    _mix_frac = _mix_frac_cfg
                if _mix_frac > 0.0 and (cur_it % _mix_every == 0):
                    _v_full = float(full_eval(program))
                    value = float((1.0 - _mix_frac) * float(value) + _mix_frac * _v_full)
                    _mix_used = True
        except Exception:
            _mix_used = False
        mix_alpha = float(getattr(args, 'true_mixin_alpha', 0.0) or 0.0)
        if mix_alpha > 0.0:
            try:
                tmap = true_holder.get('map', {})
                if prog_hash in tmap:
                    tval = float(tmap[prog_hash])
                    value = float((1.0 - mix_alpha) * float(value) + mix_alpha * tval)
            except Exception:
                pass
        # 若启用混合，则不缓存混合值，避免污染短评估缓存
        if not _mix_used:
            cache.put(cache_key, value)
        return value

    def short_eval_all(program: list):
        dur = args.short_duration
        prog_hash = hash_program(program)
        cache_key = f"{dur}|allT{len(trajectories)}|{prog_hash}|SHORT"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        ag = agent_holder.get('agent')
        if ag is not None and hasattr(ag, '_tt_salt'):
            try:
                ag._tt_salt = f"{dur}|allT{len(trajectories)}|SHORT"
            except Exception:
                pass
        scores = []
        if pool_holder['enabled']:
            packs = [(
                traj, program, dur, args.quiet_sim, args.deep_quiet,
                disturbances, reward_weights, args.log_skip, args.in_memory_log,
                bool(getattr(args, 'compose_by_gain', False)),
                getattr(args,'clip_P',None), getattr(args,'clip_I',None), getattr(args,'clip_D',None),
                float(getattr(args,'overlap_penalty',0.0) or 0.0), float(getattr(args,'conflict_penalty',0.0) or 0.0),
                getattr(args,'semantics', None), int(getattr(args,'require_k',0) or 0), int(getattr(args,'blend_topk_k',2) or 2),
                getattr(args,'gain_slew_limit', None), int(getattr(args,'min_hold_steps',0) or 0)
            ) for traj in trajectories]
            try:
                scores = pool_holder['pool'].map(_worker_evaluate_single, packs)  # type: ignore
            except Exception as _pm:
                print(f"[Parallel][WARN] 进程池执行失败(short_eval_all)，回退单进程: {_pm}")
                for traj in trajectories:
                    scores.append(_worker_evaluate_single((
                        traj, program, dur, args.quiet_sim, args.deep_quiet,
                        disturbances, reward_weights, args.log_skip, args.in_memory_log,
                        bool(getattr(args, 'compose_by_gain', False)),
                        getattr(args,'clip_P',None), getattr(args,'clip_I',None), getattr(args,'clip_D',None),
                        float(getattr(args,'overlap_penalty',0.0) or 0.0), float(getattr(args,'conflict_penalty',0.0) or 0.0),
                        getattr(args,'semantics', None), int(getattr(args,'require_k',0) or 0), int(getattr(args,'blend_topk_k',2) or 2)
                    )))
        else:
            for ti, traj in enumerate(trajectories):
                controller = PiLightSegmentedPIDController(
                    drone_model=DroneModel("cf2x"), program=program,
                    suppress_init_print=args.quiet_sim,
                    compose_by_gain=bool(getattr(args, 'compose_by_gain', False)),
                    clip_P=getattr(args,'clip_P',None), clip_I=getattr(args,'clip_I',None), clip_D=getattr(args,'clip_D',None),
                    semantics=getattr(args,'semantics', None), require_k=int(getattr(args,'require_k',0) or 0), blend_topk_k=int(getattr(args,'blend_topk_k',2) or 2),
                    gain_slew_limit=getattr(args,'gain_slew_limit', None), min_hold_steps=int(getattr(args,'min_hold_steps',0) or 0)
                )
                external_env = None
                if args.reuse_env:
                    key = (dur, ti)
                    if key not in env_pool:
                        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
                        from gym_pybullet_drones.envs.BaseAviary import Physics
                        import numpy as _np
                        init_xyz = _np.array([traj.get('initial_xyz', [0,0,0.5])])
                        env_pool[key] = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=init_xyz,
                                                   physics=Physics("pyb"), pyb_freq=240, ctrl_freq=48, gui=False, record=False)
                    external_env = env_pool[key]
                tester = SimulationTester(controller=controller,
                                          test_scenarios=disturbances,
                                          output_folder='01_pi_flight/results/mcts_eval',
                                          gui=False,
                                          weights=reward_weights,
                                          trajectory=traj,
                                          duration_sec=dur,
                                          log_skip=args.log_skip,
                                          in_memory=args.in_memory_log,
                                          external_env=external_env,
                                          reuse_reset=args.reuse_env,
                                          quiet=args.quiet_sim)
                rew = tester.run()
                try:
                    if (getattr(args,'overlap_penalty',0.0) or 0.0)>0 or (getattr(args,'conflict_penalty',0.0) or 0.0)>0:
                        metrics = controller.get_overlap_metrics()
                        mean_overlap = float(metrics.get('mean_overlap', 1.0))
                        mean_action_diff = float(metrics.get('mean_action_diff', 0.0))
                        rew = float(rew) - float(mean_overlap) * float(getattr(args,'overlap_penalty',0.0) or 0.0) - float(mean_action_diff) * float(getattr(args,'conflict_penalty',0.0) or 0.0)
                except Exception:
                    pass
                scores.append(rew)
        if args.aggregate == 'mean':
            value = float(sum(scores)/len(scores))
        elif args.aggregate == 'min':
            value = float(min(scores))
        elif args.aggregate == 'harmonic':
            import math
            value = len(scores)/sum(1/(s+1e-9) for s in scores)
        else:
            value = float(sum(scores)/len(scores))
        cache.put(cache_key, value)
        return value

    def full_eval(program: list):
        dur = args.full_duration
        prog_hash = hash_program(program)
        tv = int(true_holder.get('version', 0))
        ver_suffix = f"|TV{tv}" if float(getattr(args, 'true_mixin_alpha', 0.0) or 0.0) > 0.0 else ""
        cache_key = f"{dur}|allT{len(trajectories)}|{prog_hash}{ver_suffix}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        ag = agent_holder.get('agent')
        if ag is not None and hasattr(ag, '_tt_salt'):
            try:
                ag._tt_salt = f"{dur}|allT{len(trajectories)}"
            except Exception:
                pass
        scores = []
        if pool_holder['enabled']:
            packs = [(
                traj, program, dur, args.quiet_sim, args.deep_quiet,
                disturbances, reward_weights, args.log_skip, args.in_memory_log,
                bool(getattr(args, 'compose_by_gain', False)),
                getattr(args,'clip_P',None), getattr(args,'clip_I',None), getattr(args,'clip_D',None),
                float(getattr(args,'overlap_penalty',0.0) or 0.0), float(getattr(args,'conflict_penalty',0.0) or 0.0),
                getattr(args,'semantics', None), int(getattr(args,'require_k',0) or 0), int(getattr(args,'blend_topk_k',2) or 2),
                getattr(args,'gain_slew_limit', None), int(getattr(args,'min_hold_steps',0) or 0)
            ) for traj in trajectories]
            try:
                scores = pool_holder['pool'].map(_worker_evaluate_single, packs)  # type: ignore
            except Exception as _pm:
                print(f"[Parallel][WARN] 进程池执行失败(full_eval)，回退单进程: {_pm}")
                for traj in trajectories:
                    scores.append(_worker_evaluate_single((
                        traj, program, dur, args.quiet_sim, args.deep_quiet,
                        disturbances, reward_weights, args.log_skip, args.in_memory_log,
                        bool(getattr(args, 'compose_by_gain', False)),
                        getattr(args,'clip_P',None), getattr(args,'clip_I',None), getattr(args,'clip_D',None),
                        float(getattr(args,'overlap_penalty',0.0) or 0.0), float(getattr(args,'conflict_penalty',0.0) or 0.0),
                        getattr(args,'semantics', None), int(getattr(args,'require_k',0) or 0), int(getattr(args,'blend_topk_k',2) or 2)
                    )))
        else:
            for ti, traj in enumerate(trajectories):
                controller = PiLightSegmentedPIDController(
                    drone_model=DroneModel("cf2x"), program=program,
                    suppress_init_print=args.quiet_sim,
                    compose_by_gain=bool(getattr(args, 'compose_by_gain', False)),
                    clip_P=getattr(args,'clip_P',None), clip_I=getattr(args,'clip_I',None), clip_D=getattr(args,'clip_D',None),
                    semantics=getattr(args,'semantics', None), require_k=int(getattr(args,'require_k',0) or 0), blend_topk_k=int(getattr(args,'blend_topk_k',2) or 2),
                    gain_slew_limit=getattr(args,'gain_slew_limit', None), min_hold_steps=int(getattr(args,'min_hold_steps',0) or 0)
                )
                external_env = None
                if args.reuse_env:
                    key = (dur, ti)
                    if key not in env_pool:
                        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
                        from gym_pybullet_drones.envs.BaseAviary import Physics
                        import numpy as _np
                        init_xyz = _np.array([traj.get('initial_xyz', [0,0,0.5])])
                        env_pool[key] = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=init_xyz,
                                                   physics=Physics("pyb"), pyb_freq=240, ctrl_freq=48, gui=False, record=False)
                    external_env = env_pool[key]
                tester = SimulationTester(controller=controller,
                                          test_scenarios=disturbances,
                                          output_folder='01_pi_flight/results/mcts_eval',
                                          gui=False,
                                          weights=reward_weights,
                                          trajectory=traj,
                                          duration_sec=dur,
                                          log_skip=args.log_skip,
                                          in_memory=args.in_memory_log,
                                          external_env=external_env,
                                          reuse_reset=args.reuse_env,
                                          quiet=args.quiet_sim)
                rew = tester.run()
                try:
                    if (getattr(args,'overlap_penalty',0.0) or 0.0)>0 or (getattr(args,'conflict_penalty',0.0) or 0.0)>0:
                        metrics = controller.get_overlap_metrics()
                        mean_overlap = float(metrics.get('mean_overlap', 1.0))
                        mean_action_diff = float(metrics.get('mean_action_diff', 0.0))
                        rew = float(rew) - float(mean_overlap) * float(getattr(args,'overlap_penalty',0.0) or 0.0) - float(mean_action_diff) * float(getattr(args,'conflict_penalty',0.0) or 0.0)
                except Exception:
                    pass
                scores.append(rew)
        if args.aggregate == 'mean':
            value = float(sum(scores)/len(scores))
        elif args.aggregate == 'min':
            value = float(min(scores))
        elif args.aggregate == 'harmonic':
            import math
            value = len(scores)/sum(1/(s+1e-9) for s in scores)
        else:
            value = float(sum(scores)/len(scores))
        mix_alpha = float(getattr(args, 'true_mixin_alpha', 0.0) or 0.0)
        if mix_alpha > 0.0:
            try:
                tmap = true_holder.get('map', {})
                if prog_hash in tmap:
                    tval = float(tmap[prog_hash])
                    value = float((1.0 - mix_alpha) * float(value) + mix_alpha * tval)
            except Exception:
                pass
        cache.put(cache_key, value)
        return value

    def full_eval_true(program: list):
        dur = args.full_duration
        prog_hash = hash_program(program)
        cache_key = f"{dur}|allT{len(trajectories)}|{prog_hash}|TRUE"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        ag = agent_holder.get('agent')
        if ag is not None and hasattr(ag, '_tt_salt'):
            try:
                ag._tt_salt = f"{dur}|allT{len(trajectories)}|TRUE"
            except Exception:
                pass
        scores = []
        for ti, traj in enumerate(trajectories):
            controller = PiLightSegmentedPIDController(
                drone_model=DroneModel("cf2x"), program=program,
                suppress_init_print=args.quiet_sim,
                compose_by_gain=bool(getattr(args, 'compose_by_gain', False)),
                clip_P=getattr(args,'clip_P',None), clip_I=getattr(args,'clip_I',None), clip_D=getattr(args,'clip_D',None),
                semantics=getattr(args,'semantics', None), require_k=int(getattr(args,'require_k',0) or 0), blend_topk_k=int(getattr(args,'blend_topk_k',2) or 2)
            )
            external_env = None
            if args.reuse_env:
                key = (dur, ti)
                if key not in env_pool:
                    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
                    from gym_pybullet_drones.envs.BaseAviary import Physics
                    import numpy as _np
                    init_xyz = _np.array([traj.get('initial_xyz', [0,0,0.5])])
                    env_pool[key] = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=init_xyz,
                                               physics=Physics("pyb"), pyb_freq=240, ctrl_freq=48, gui=False, record=False)
                external_env = env_pool[key]
            tester = SimulationTester(controller=controller,
                                      test_scenarios=disturbances,
                                      output_folder='01_pi_flight/results/mcts_eval',
                                      gui=False,
                                      weights=reward_weights,
                                      trajectory=traj,
                                      duration_sec=dur,
                                      log_skip=args.log_skip,
                                      in_memory=args.in_memory_log,
                                      external_env=external_env,
                                      reuse_reset=args.reuse_env,
                                      quiet=args.quiet_sim)
            rew = tester.run()
            scores.append(float(rew))
        if args.aggregate == 'mean':
            value = float(sum(scores)/len(scores))
        elif args.aggregate == 'min':
            value = float(min(scores))
        elif args.aggregate == 'harmonic':
            import math
            value = len(scores)/sum(1/(s+1e-9) for s in scores)
        else:
            value = float(sum(scores)/len(scores))
        cache.put(cache_key, value)
        return value

    # mini_full_eval 与 TR 相关逻辑已移除，简化训练流程

    evaluation_func = dynamic_eval

    warm_prog = None
    if getattr(args, 'warm_start_cmaes', False):
        try:
            with open(args.warm_start_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'rules' in data:
                try:
                    prog = deserialize_program(data)
                    if isinstance(prog, list) and len(prog) > 0:
                        warm_prog = prog
                        kp = ki = kd = None
                        try:
                            for a in prog[0].get('action', []):
                                if isinstance(a, BinaryOpNode) and a.op == 'set' and isinstance(a.left, TerminalNode) and isinstance(a.right, TerminalNode):
                                    if a.left.value == 'P': kp = float(a.right.value)
                                    elif a.left.value == 'I': ki = float(a.right.value)
                                    elif a.left.value == 'D': kd = float(a.right.value)
                        except Exception:
                            pass
                        if all(v is not None for v in (kp,ki,kd)):
                            print(f"[WarmStart] 载入 best_program (CMA-ES) -> P={kp:.4f} I={ki:.4f} D={kd:.4f}")
                        else:
                            print("[WarmStart] 载入 best_program (CMA-ES) -> 单规则已注入")
                    else:
                        print('[WarmStart][WARN] best_program JSON 中 rules 为空，跳过 warm start')
                except Exception as _dp_e:
                    print(f"[WarmStart][WARN] 解析 best_program JSON 失败，将尝试 legacy 格式: {_dp_e}")
            if warm_prog is None:
                params = data.get('best_params') or data.get('best_params'.upper())
                if isinstance(params, list) and len(params) >= 3:
                    kp, ki, kd = params[:3]
                    condition = BinaryOpNode('>', TerminalNode('pos_err_x'), TerminalNode(-999.0))
                    action = [
                        BinaryOpNode('set', TerminalNode('P'), TerminalNode(round(float(kp),4))),
                        BinaryOpNode('set', TerminalNode('I'), TerminalNode(round(float(ki),4))),
                        BinaryOpNode('set', TerminalNode('D'), TerminalNode(round(float(kd),4)))
                    ]
                    warm_prog = [{'condition': condition, 'action': action}]
                    print(f"[WarmStart] Loaded CMA-ES gains P={kp:.4f} I={ki:.4f} D={kd:.4f} -> 注入初始程序")
                else:
                    print('[WarmStart][WARN] 未找到 best_params，跳过 warm start')
        except Exception as e:
            print(f'[WarmStart][ERROR] 读取 {args.warm_start_path} 失败: {e}')
    if getattr(args, 'warm_start_program', None):
        try:
            pj = str(getattr(args, 'warm_start_program'))
            with open(pj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'rules' in data:
                warm_prog = deserialize_program(data)
                print(f"[WarmStart] 从 PI-Flight 程序载入: {pj} -> 规则数={len(warm_prog) if isinstance(warm_prog, list) else 'N/A'}")
            else:
                print(f"[WarmStart][WARN] {pj} 不是包含 'rules' 的程序 JSON，忽略")
        except Exception as e:
            print(f"[WarmStart][ERROR] 加载 --warm-start-program 失败: {e}")

    def _temp_eval(prog:list):
        scores=[]
        for traj in trajectories:
            controller = PiLightSegmentedPIDController(
                drone_model=DroneModel("cf2x"), program=prog,
                suppress_init_print=args.quiet_sim,
                compose_by_gain=bool(getattr(args, 'compose_by_gain', False)),
                clip_P=getattr(args,'clip_P',None), clip_I=getattr(args,'clip_I',None), clip_D=getattr(args,'clip_D',None),
                semantics=getattr(args,'semantics', None), require_k=int(getattr(args,'require_k',0) or 0), blend_topk_k=int(getattr(args,'blend_topk_k',2) or 2),
                gain_slew_limit=getattr(args,'gain_slew_limit', None), min_hold_steps=int(getattr(args,'min_hold_steps',0) or 0)
            )
            tester = SimulationTester(controller=controller,
                                      test_scenarios=disturbances,
                                      output_folder='01_pi_flight/results/mcts_eval',
                                      gui=False,
                                      weights=reward_weights,
                                      trajectory=traj,
                                      duration_sec=args.full_duration,
                                      log_skip=args.log_skip,
                                      in_memory=args.in_memory_log,
                                      quiet=args.quiet_sim)
            scores.append(tester.run())
        if args.aggregate == 'mean':
            return float(sum(scores)/len(scores))
        elif args.aggregate == 'min':
            return float(min(scores))
        elif args.aggregate == 'harmonic':
            import math
            return len(scores)/sum(1/(s+1e-9) for s in scores)
        return float(sum(scores)/len(scores))

    # CMA-Joint 相关工具与逻辑已删除，统一由 MCTS 搜索与 ML 调度负责

    base_complexity = 0.0
    print('[Config] 复杂度惩罚：已全局关闭 (complexity_penalty=0)')

    agent = MCTS_Agent(
        _temp_eval,
        DSL_VARIABLES,
        DSL_CONSTANTS,
        DSL_OPERATORS,
        max_depth=args.mcts_max_depth,
        rollout_depth=args.rollout_depth,
        complexity_penalty=base_complexity,
        pw_alpha=args.pw_alpha,
        pw_c=args.pw_c,
        warm_start_program=warm_prog
    )
    agent._complexity_min_scale = 0.0
    agent._complexity_max_scale = 0.0
    agent._complexity_ramp_start = 0.0
    agent._complexity_ramp_end = 0.0
    if hasattr(agent, '_dynamic_complexity'):
        agent._dynamic_complexity = 0.0
    if hasattr(agent, '_min_rules_guard'):
        agent._min_rules_guard = max(1, int(args.min_rules_guard))
    if hasattr(agent, '_max_rules'):
        agent._max_rules = max(2, int(args.max_rules))
    if hasattr(agent, '_add_rule_bias_base'):
        agent._add_rule_bias_base = max(1, int(args.add_rule_bias_base))

    try:
        if hasattr(agent, '_epsilon_max'):
            agent._epsilon_max = float(args.epsilon_max)
        if hasattr(agent, '_epsilon_end_progress'):
            agent._epsilon_end_progress = float(args.epsilon_end_progress)
        if hasattr(agent, '_swap_span'):
            agent._swap_span = int(args.swap_span)
        if hasattr(agent, '_stagnation_window'):
            agent._stagnation_window = int(args.stagnation_window)
        if hasattr(agent, '_epsilon_rebound'):
            agent._epsilon_rebound = float(args.epsilon_rebound)
        if hasattr(agent, '_rebound_iters'):
            agent._rebound_iters = int(args.rebound_iters)
        if hasattr(agent, '_diversity_bonus_max'):
            agent._diversity_bonus_max = float(args.diversity_bonus_max)
        if hasattr(agent, '_diversity_end_progress'):
            agent._diversity_end_progress = float(args.diversity_end_progress)
        if hasattr(agent, '_strict_bonus_scale'):
            agent._strict_bonus_scale = float(args.strict_bonus_scale)
        if hasattr(agent, '_prefer_more_rules_tie_delta'):
            agent._prefer_more_rules_tie_delta = float(getattr(args, 'prefer_more_rules_tie_delta', 0.0))
        if hasattr(agent, '_prefer_fewer_rules_tie_delta'):
            agent._prefer_fewer_rules_tie_delta = float(getattr(args, 'prefer_fewer_rules_tie_delta', 0.0))
        if hasattr(agent, '_full_action_prob'):
            agent._full_action_prob = float(getattr(args, 'full_action_prob', 0.0))
        if hasattr(agent, '_allowed_cond_unaries'):
            try:
                allow_raw = getattr(args, 'allowed_cond_unaries', 'identity,abs') or 'identity,abs'
                allow_set = set([s.strip() for s in allow_raw.split(',') if s.strip()])
                if 'identity' not in allow_set:
                    allow_set.add('identity')
                agent._allowed_cond_unaries = allow_set
                print(f"[DSL] allowed cond unaries: {sorted(list(agent._allowed_cond_unaries))}")
            except Exception as _au_e:
                print(f"[DSL][WARN] 解析 --allowed-cond-unaries 失败，使用默认: {_au_e}")
        if hasattr(agent, '_trig_as_phase_window'):
            agent._trig_as_phase_window = bool(getattr(args, 'trig_as_phase_window', False))
        if hasattr(agent, '_trig_lt_max'):
            agent._trig_lt_max = float(getattr(args, 'trig_lt_max', 0.25))
        if hasattr(agent, '_enable_macros'):
            agent._enable_macros = bool(getattr(args, 'enable_macros', False))
        if hasattr(agent, '_edit_credit_mode'):
            agent._edit_credit_mode = str(getattr(args, 'edit_credit', 'off') or 'off')
        if hasattr(agent, '_edit_credit_c'):
            agent._edit_credit_c = float(getattr(args, 'edit_credit_c', 0.8) or 0.8)
        # MCTS 先验（NN/启发式）与手动注入已移除，统一由 ML 层调参驱动
    except Exception as _exp_e:
        print(f"[Explore][WARN] 注入 epsilon/swap 参数失败: {_exp_e}")

    try:
        if hasattr(agent, '_min_rules_guard_initial'):
            agent._min_rules_guard_initial = int(agent._min_rules_guard)
        if hasattr(agent, '_min_rules_guard_final'):
            if args.min_rules_final is not None:
                agent._min_rules_guard_final = max(1, int(args.min_rules_final))
            else:
                agent._min_rules_guard_final = max(1, int(agent._min_rules_guard))
        if hasattr(agent, '_min_rules_ramp_start'):
            agent._min_rules_ramp_start = float(args.min_rules_ramp_start)
        if hasattr(agent, '_min_rules_ramp_end'):
            agent._min_rules_ramp_end = float(args.min_rules_ramp_end)
        if hasattr(agent, '_min_rules_guard_effective'):
            agent._min_rules_guard_effective = int(agent._min_rules_guard)
    except Exception as _dyn_min_e:
        print(f"[MinRules][WARN] 注入动态下限参数失败: {_dyn_min_e}")

    try:
        if hasattr(agent, 'root') and hasattr(agent, '_min_rules_guard'):
            pad_ok = True
            if warm_prog is not None and not getattr(args, 'pad_after_warm_start', False):
                pad_ok = False
            if hasattr(agent, '_pad_after_warm_start'):
                agent._pad_after_warm_start = bool(getattr(args, 'pad_after_warm_start', False))
            if pad_ok:
                cur_rules = len(agent.root.program) if agent.root and hasattr(agent.root, 'program') else 0
                while cur_rules < agent._min_rules_guard and cur_rules < getattr(agent, '_max_rules', 8):
                    agent.root.program.append(agent._generate_random_rule())
                    cur_rules += 1
            else:
                print('[WarmStart] 保持单分段，不在 warm start 后强行补齐到 min-rules-guard')
    except Exception as _init_seg_e:
        print(f"[InitSeg][WARN] 补齐最小分段失败: {_init_seg_e}")

    agent_holder['agent'] = agent
    agent.evaluation_function = dynamic_eval

    # --- ML 独占模式：锁定 MCTS 超参为稳健默认，由 ML 持有 ---
    if bool(getattr(args, 'ml_exclusive', False)):
        try:
            if hasattr(agent, '_puct_enable'):
                agent._puct_enable = True
            if hasattr(agent, '_puct_c'):
                agent._puct_c = 1.0
            if hasattr(agent, '_dirichlet_eps'):
                agent._dirichlet_eps = 0.20
            if hasattr(agent, '_dirichlet_alpha'):
                agent._dirichlet_alpha = 0.3
            if hasattr(agent, '_value_mix_lambda'):
                agent._value_mix_lambda = 0.10
            agent.pw_alpha = 0.82
            agent.pw_c = 1.10
            if hasattr(agent, '_add_rule_bias_base'):
                agent._add_rule_bias_base = max(6, int(getattr(agent, '_add_rule_bias_base', 2)))
            if hasattr(agent, '_full_action_prob'):
                agent._full_action_prob = max(0.55, float(getattr(agent, '_full_action_prob', 0.0)))
            if hasattr(agent, '_prefer_more_rules_tie_delta'):
                agent._prefer_more_rules_tie_delta = max(0.02, float(getattr(agent, '_prefer_more_rules_tie_delta', 0.0)))
            if hasattr(agent, '_prefer_fewer_rules_tie_delta'):
                agent._prefer_fewer_rules_tie_delta = 0.0
            if hasattr(agent, '_epsilon_max'):
                agent._epsilon_max = min(0.25, float(getattr(agent, '_epsilon_max', 0.25)))
            print('[ML-Exclusive] AlphaZero-lite defaults applied. Manual MCTS knobs are ignored; ML layer owns them.')
        except Exception as _mx_e:
            print(f"[ML-Exclusive][WARN] 初始化默认失败：{_mx_e}")

    # --- ML scheduler wiring (optional) ---
    ml_sched = None
    ml_allowed = set([s.strip() for s in str(getattr(args,'ml_allowed','')).split(',') if s.strip()])
    try:
        from .ml_param_scheduler import HeuristicScheduler, NNScheduler, MCTSContext, parse_bounds_spec, apply_mcts_param_updates  # type: ignore
    except Exception:
        # Fallback 1: direct import from current package directory (script mode)
        try:
            import ml_param_scheduler as _mls  # type: ignore
            HeuristicScheduler = getattr(_mls, 'HeuristicScheduler')
            NNScheduler = getattr(_mls, 'NNScheduler')
            MCTSContext = getattr(_mls, 'MCTSContext')
            parse_bounds_spec = getattr(_mls, 'parse_bounds_spec')
            apply_mcts_param_updates = getattr(_mls, 'apply_mcts_param_updates')
        except Exception:
            # Fallback 2: dynamic package alias created earlier (_PKG_NAME)
            import importlib as _il
            _mod = _il.import_module(f"{_PKG_NAME}.ml_param_scheduler")  # type: ignore[name-defined]
            HeuristicScheduler = getattr(_mod, 'HeuristicScheduler')
            NNScheduler = getattr(_mod, 'NNScheduler')
            MCTSContext = getattr(_mod, 'MCTSContext')
            parse_bounds_spec = getattr(_mod, 'parse_bounds_spec')
            apply_mcts_param_updates = getattr(_mod, 'apply_mcts_param_updates')
    safe_bounds = parse_bounds_spec(getattr(args,'ml_safe_bounds',''))
    if str(getattr(args,'ml_scheduler','none')) != 'none':
        if str(getattr(args,'ml_scheduler')) == 'heuristic':
            ml_sched = HeuristicScheduler(strategy=str(getattr(args,'ml_strategy','absolute')), allowed=ml_allowed, safe_bounds=safe_bounds, log=bool(getattr(args,'ml_log', False)))
        elif str(getattr(args,'ml_scheduler')) == 'nn':
            ml_sched = NNScheduler(model_path=str(getattr(args,'ml_path','') or ''), strategy=str(getattr(args,'ml_strategy','absolute')), allowed=ml_allowed, safe_bounds=safe_bounds, log=bool(getattr(args,'ml_log', False)))

    # AlphaZero-lite 手动注入选项已移除，默认交由 ML 层/独占模式管理

    # 在线策略先验训练与采样钩子已移除，避免额外依赖

    # 复核/门控与 TestVerify 已移除，统一以训练集 best 为准

    print(f"[Perf] short={args.short_duration}s full={args.full_duration}s short_frac={args.short_frac:.2f} log_skip={args.log_skip} in_memory={args.in_memory_log}")
    print(f"[MCTS] Config: max_depth={args.mcts_max_depth} rollout_depth={args.rollout_depth} complexity_penalty={args.complexity_penalty} pw_alpha={args.pw_alpha} pw_c={args.pw_c}")
    if pool_holder['enabled']:
        print(f"[Parallel] Using persistent Pool: workers={pool_holder['worker_n']} | traj_batch_size={traj_batch_size}/{T}")
    else:
        print(f"[Parallel] Single-process eval | traj_batch_size={traj_batch_size}/{T}")
    total = args.iters
    report_interval = max(1, total // args.report) if args.report>0 else max(1,total//20)
    t0 = time.time()
    last_improve_time = t0
    last_best_reward_seen = -1e18
    ckpt_last_saved_train = -1e18
    print(f"[INFO] Multi-trajectory search: {traj_names} | disturbances={bool(disturbances)} aggregate={args.aggregate}")
    print(f"[INFO] For long run (e.g., 30000 iters) consider: lower --duration or subset trajectories per iter for speed.")

    use_tqdm = False
    if args.tqdm:
        try:
            from tqdm import trange  # type: ignore
            tqdm_range = trange(1, total+1, desc='MCTS', ncols=100, leave=True)  # type: ignore
            iter_range = tqdm_range  # type: ignore
            use_tqdm = True
        except Exception:
            print('[WARN] 未能导入 tqdm，改用普通日志。')
            iter_range = range(1, total+1)
    else:
        iter_range = range(1, total+1)

    import contextlib, io
    suppress = args.quiet_eval
    last_verified_hash = None
    log_best_train_hash = None
    log_best_train_score = float('nan')
    saved_penalty = {
        'overlap': float(getattr(args, 'overlap_penalty', 0.0) or 0.0),
        'conflict': float(getattr(args, 'conflict_penalty', 0.0) or 0.0),
        'complexity': float(base_complexity),
    }
    no_improve_count = 0
    unfreeze_until_iter = 0
    burst_since_size = 0
    last_prog_size_seen = 0
    burst_until_iter = 0
    burst_saved: Dict[str, Any] = {
        'min_rules_guard_effective': None,
        'add_rule_bias_base': None,
        'prefer_more_rules_tie_delta': None,
    }

    # 初始验证、全局最优注入与测试集验证已移除

    # ML 调度的改进跟踪初值（避免未绑定）
    last_best_reward_for_ml = -1e18
    last_improve_time_for_ml = t0
    last_improve_iter_for_ml = 0

    # 预设循环外变量，避免在极端情况下未绑定（如 0 次迭代）
    it = 0
    best_prog = None
    best_reward_now = float('nan')
    prog_size = 0

    for it in iter_range:
        if getattr(args, 'banner_every', 0) and args.banner_every > 0 and ((it-1) % args.banner_every == 0):
            banner = f"{'-'*18}第{it}/{total}次迭代{'-'*12}"
            if use_tqdm:
                try:
                    from tqdm import tqdm  # type: ignore
                    tqdm.write(banner)
                except Exception:
                    print(banner)
            else:
                print(banner)
        if suppress:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                agent.search(iterations=1, total_target=total)
        else:
            agent.search(iterations=1, total_target=total)

        if use_tqdm:
            if it % report_interval == 0 or it == 1:
                elapsed = time.time() - t0
                speed = it/elapsed if elapsed>0 else 0.0
                remaining = total - it
                eta = remaining/speed if speed>0 else float('inf')
                _, bs = agent.get_best_program()
                try:
                    tqdm_range.set_postfix(best=f"{bs:.4f}", it_s=f"{speed:.2f}", ETA=f"{eta/60:.1f}m")  # type: ignore
                except Exception:
                    pass
        else:
            if it % report_interval == 0 or it == 1:
                elapsed = time.time() - t0
                speed = it/elapsed if elapsed>0 else 0.0
                remaining = total - it
                eta = remaining/speed if speed>0 else float('inf')
                _, bs = agent.get_best_program()
                print(f"[MCTS] {it}/{total} best={bs:.4f} speed={speed:.2f} it/s ETA={eta:.1f}s")

        # ML scheduler hook (post-search for this iter)
        try:
            # Respect explicit 0: don't coerce 0 to default via `or`.
            _warm_raw = getattr(args, 'ml_warmup_iters', 10)
            try:
                warmup_iters = int(_warm_raw)
            except Exception:
                warmup_iters = 10
            _intv_raw = getattr(args, 'ml_interval', 5)
            try:
                interval = int(_intv_raw)
            except Exception:
                interval = 5
            interval = max(1, interval)
            if ml_sched is not None and it >= warmup_iters:
                if (it % interval) == 0:
                    # Build context
                    try:
                        best_prog, best_val = agent.get_best_program()
                    except Exception:
                        best_prog = None
                        best_val = float('nan')
                    # derive improvement signals
                    delta = 0.0
                    if isinstance(best_val, (int, float)) and best_val > last_best_reward_for_ml + 1e-12:
                        delta = float(best_val - last_best_reward_for_ml)
                        last_best_reward_for_ml = float(best_val)
                        last_improve_time_for_ml = time.time()
                        last_improve_iter_for_ml = int(it)
                    sec_since = time.time() - (last_improve_time_for_ml if 'last_improve_time_for_ml' in locals() else t0)
                    it_since = it - (last_improve_iter_for_ml if 'last_improve_iter_for_ml' in locals() else 0)
                    rule_count = len(best_prog) if best_prog else 0
                    ctx_obj = MCTSContext(
                        iter_idx=int(it),
                        total_target=int(total),
                        progress=float(it/max(1,total)),
                        best_reward=float(best_val if isinstance(best_val,(int,float)) else float('nan')),
                        best_reward_delta=float(delta),
                        seconds_since_improve=float(sec_since if isinstance(sec_since,(int,float)) else -1.0),
                        iters_since_improve=int(it_since),
                        rule_count=int(rule_count),
                        epsilon=float(getattr(agent,'epsilon',0.0) or 0.0),
                        stagnation_window=int(getattr(agent,'_stagnation_window',0) or 0)
                    )
                    updates = ml_sched.step(ctx_obj)
                    # Optionally dump feature->target pair for NN training
                    dump_path = str(getattr(args, 'ml_dump_csv', '') or '').strip()
                    try:
                        if dump_path:
                            import csv
                            import os
                            # Import shared KEY_ORDER for stable column names (package + script modes)
                            try:
                                from .ml_param_scheduler import KEY_ORDER  # type: ignore
                            except Exception:
                                try:
                                    from ml_param_scheduler import KEY_ORDER  # type: ignore
                                except Exception:
                                    try:
                                        import importlib as _il_k
                                        KEY_ORDER = getattr(_il_k.import_module(f"{_PKG_NAME}.ml_param_scheduler"), 'KEY_ORDER')  # type: ignore[name-defined]
                                    except Exception:
                                        # Last-resort fallback to a local constant to avoid breaking CSV dump
                                        KEY_ORDER = [
                                            'pw_alpha','pw_c','_puct_c','_edit_prior_c','_dirichlet_eps','_full_action_prob',
                                            '_prefer_more_rules_tie_delta','_prefer_fewer_rules_tie_delta','_add_rule_bias_base',
                                            '_value_mix_lambda','_epsilon_max'
                                        ]
                            # Keep features aligned with NNScheduler input (7 dims)
                            cols_feat = [
                                'progress','best_reward','best_reward_delta',
                                'seconds_since_improve','iters_since_improve','rule_count','epsilon'
                            ]
                            header = cols_feat + KEY_ORDER
                            os.makedirs(os.path.dirname(dump_path) or '.', exist_ok=True)
                            new_file = not os.path.exists(dump_path)
                            row_feat = [
                                float(ctx_obj.progress),
                                float(ctx_obj.best_reward), float(ctx_obj.best_reward_delta),
                                float(ctx_obj.seconds_since_improve), int(ctx_obj.iters_since_improve),
                                int(ctx_obj.rule_count), float(ctx_obj.epsilon)
                            ]
                            # targets: fall back to current agent values if scheduler returned empty or partial
                            target_vals = []
                            for k in KEY_ORDER:
                                if updates and (k in updates):
                                    target_vals.append(float(updates[k]))
                                else:
                                    try:
                                        # Fallback to agent's current value, with a safe default if missing
                                        target_vals.append(float(getattr(agent, k, float('nan'))))
                                    except Exception:
                                        target_vals.append(float('nan'))
                            row = row_feat + target_vals
                            with open(dump_path, 'a', newline='', encoding='utf-8') as fcsv:
                                w = csv.writer(fcsv)
                                if new_file:
                                    w.writerow(header)
                                w.writerow(row)
                                try:
                                    fcsv.flush()
                                    import os as _osfs
                                    try:
                                        _osfs.fsync(fcsv.fileno())
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            try:
                                if it % interval == 0:
                                    print(f"[ML-Sched] Dumped CSV row -> {dump_path} @ iter {it}")
                            except Exception:
                                pass
                    except Exception as _dump_e:
                        try:
                            _dp = dump_path if 'dump_path' in locals() and dump_path else '(unset)'
                            print(f"[ML-Sched][WARN] 样本导出失败（iter={it} path={_dp}）: {_dump_e}")
                        except Exception:
                            pass
                    if updates:
                        apply_mcts_param_updates(agent, updates,
                                                 strategy=str(getattr(args,'ml_strategy','absolute')),
                                                 bounds=safe_bounds,
                                                 int_keys={'_add_rule_bias_base'},
                                                 log=bool(getattr(args,'ml_log', False)))
        except Exception as _ml_e:
            if it == 1:
                print(f"[ML-Sched][WARN] 调度器异常（已忽略）：{_ml_e}")

        # 轻量候补优化已移除，避免分支膨胀

        try:
            cur_prog, cur_best = agent.get_best_program()
            if cur_best is None:
                cur_best = -1e18
            if cur_best > last_best_reward_seen + 1e-9:
                last_best_reward_seen = cur_best
                last_improve_time = time.time()
            else:
                stg_sec = int(getattr(args, 'stagnation_seconds', 0))
                eps_target = float(getattr(args, 'epsilon_rebound_target', 0.0) or 0.0)
                if stg_sec > 0 and eps_target > 0.0:
                    if (time.time() - last_improve_time) >= stg_sec:
                        setattr(agent, '_epsilon_rebound', max(float(getattr(agent, '_epsilon_rebound', 0.0)), eps_target))
                        until_iter = int(getattr(agent, 'total_iterations_done', it)) + (int(getattr(args, 'time_rebound_iters', 0)) or int(getattr(args, 'rebound_iters', 80)))
                        setattr(agent, '_rebound_until_iter', until_iter)
                        msg = f"[Rebound-Time] no-improve >= {stg_sec}s -> epsilon>= {eps_target:.2f} for {until_iter - int(getattr(agent, 'total_iterations_done', it))} iters"
                        if use_tqdm:
                            try:
                                from tqdm import tqdm  # type: ignore
                                tqdm.write(msg)
                            except Exception:
                                print(msg)
                        else:
                            print(msg)
                        last_improve_time = time.time()
        except Exception as _stg_e:
            if it == 1:
                print(f"[Stagnation][WARN] 时间停滞检测失败: {_stg_e}")

        # mini-full 探针评估已移除

        # 验证与自动解冻逻辑已移除

        # 在线策略先验更新已移除

        # CMA-ES 参数微调逻辑（MCTS负责结构，CMA负责参数）
        if args.cma_refine_every > 0 and it % args.cma_refine_every == 0:
            try:
                cur_prog_cma, cur_best_cma = agent.get_best_program()
                if cur_prog_cma and len(cur_prog_cma) > 0:
                    msg_start = f"[CMA-Refine] @ iter {it}: 开始微调当前最优程序（{len(cur_prog_cma)}规则）的PID参数..."
                    if use_tqdm:
                        try:
                            from tqdm import tqdm
                            tqdm.write(msg_start)
                        except Exception:
                            print(msg_start)
                    else:
                        print(msg_start)
                    
                    # 提取当前所有规则的PID参数
                    rule_params = []  # [(rule_idx, P, I, D), ...]
                    for rule_idx, rule in enumerate(cur_prog_cma):
                        kp = ki = kd = None
                        try:
                            for act in rule.get('action', []):
                                if isinstance(act, BinaryOpNode) and act.op == 'set' and isinstance(act.left, TerminalNode) and isinstance(act.right, TerminalNode):
                                    if act.left.value == 'P': kp = float(act.right.value)
                                    elif act.left.value == 'I': ki = float(act.right.value)
                                    elif act.left.value == 'D': kd = float(act.right.value)
                        except Exception:
                            pass
                        if all(v is not None for v in (kp, ki, kd)):
                            rule_params.append((rule_idx, kp, ki, kd))
                    
                    if rule_params:
                        # 导入CMA-ES
                        try:
                            import cma
                        except ImportError:
                            print("[CMA-Refine][WARN] 未安装cma库，跳过CMA微调")
                            rule_params = []
                        
                        if rule_params:
                            # 准备优化目标函数
                            def cma_objective(flat_params):
                                # flat_params: [P1, I1, D1, P2, I2, D2, ...]
                                # 重建程序
                                prog_copy = [dict(r) for r in cur_prog_cma]
                                param_idx = 0
                                for rule_idx, _, _, _ in rule_params:
                                    new_p = flat_params[param_idx]
                                    new_i = flat_params[param_idx + 1]
                                    new_d = flat_params[param_idx + 2]
                                    param_idx += 3
                                    
                                    # 更新action中的PID
                                    new_actions = []
                                    for act in prog_copy[rule_idx].get('action', []):
                                        if isinstance(act, BinaryOpNode) and act.op == 'set' and isinstance(act.left, TerminalNode):
                                            if act.left.value == 'P':
                                                new_actions.append(BinaryOpNode('set', TerminalNode('P'), TerminalNode(float(new_p))))
                                            elif act.left.value == 'I':
                                                new_actions.append(BinaryOpNode('set', TerminalNode('I'), TerminalNode(float(new_i))))
                                            elif act.left.value == 'D':
                                                new_actions.append(BinaryOpNode('set', TerminalNode('D'), TerminalNode(float(new_d))))
                                            else:
                                                new_actions.append(act)
                                        else:
                                            new_actions.append(act)
                                    prog_copy[rule_idx]['action'] = new_actions
                                
                                # 评估（使用短时长以加速）
                                try:
                                    score = float(short_eval_all(prog_copy))
                                    return -score  # CMA最小化，所以取负
                                except Exception:
                                    return 1e18
                            
                            # 构建初始参数向量
                            x0 = []
                            for _, kp, ki, kd in rule_params:
                                x0.extend([kp, ki, kd])
                            
                            # 运行CMA-ES
                            sigma0 = args.cma_sigma
                            opts = {
                                'popsize': args.cma_popsize,
                                'maxiter': args.cma_maxiter,
                                'verb_disp': 0,
                                'verb_log': 0,
                                'verbose': -9,
                            }
                            
                            try:
                                es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
                                cma_iter = 0
                                while not es.stop() and cma_iter < args.cma_maxiter:
                                    solutions = es.ask()
                                    fitness = [cma_objective(sol) for sol in solutions]
                                    es.tell(solutions, fitness)
                                    cma_iter += 1
                                
                                # 获取最优解
                                best_sol = es.result.xbest
                                best_fit = -es.result.fbest  # 转回正分数
                                
                                # 更新程序
                                param_idx = 0
                                for rule_idx, _, _, _ in rule_params:
                                    new_p = best_sol[param_idx]
                                    new_i = best_sol[param_idx + 1]
                                    new_d = best_sol[param_idx + 2]
                                    param_idx += 3
                                    
                                    # 更新当前程序的action
                                    new_actions = []
                                    for act in cur_prog_cma[rule_idx].get('action', []):
                                        if isinstance(act, BinaryOpNode) and act.op == 'set' and isinstance(act.left, TerminalNode):
                                            if act.left.value == 'P':
                                                new_actions.append(BinaryOpNode('set', TerminalNode('P'), TerminalNode(round(float(new_p), 4))))
                                            elif act.left.value == 'I':
                                                new_actions.append(BinaryOpNode('set', TerminalNode('I'), TerminalNode(round(float(new_i), 4))))
                                            elif act.left.value == 'D':
                                                new_actions.append(BinaryOpNode('set', TerminalNode('D'), TerminalNode(round(float(new_d), 4))))
                                            else:
                                                new_actions.append(act)
                                        else:
                                            new_actions.append(act)
                                    cur_prog_cma[rule_idx]['action'] = new_actions
                                
                                # 验证改进并注入回MCTS
                                refined_score = float(short_eval_all(cur_prog_cma))
                                if refined_score > cur_best_cma + 1e-6:
                                    # 注入回MCTS的best
                                    try:
                                        agent.best_program = cur_prog_cma
                                        agent.best_value = refined_score
                                        msg_success = f"[CMA-Refine] 完成! 得分提升: {cur_best_cma:.4f} -> {refined_score:.4f} (+{refined_score - cur_best_cma:.4f})"
                                    except Exception:
                                        msg_success = f"[CMA-Refine] 完成! 得分: {refined_score:.4f} (无法注入MCTS，仅记录)"
                                else:
                                    msg_success = f"[CMA-Refine] 完成! 得分未提升: {cur_best_cma:.4f} -> {refined_score:.4f}"
                                
                                if use_tqdm:
                                    try:
                                        from tqdm import tqdm
                                        tqdm.write(msg_success)
                                    except Exception:
                                        print(msg_success)
                                else:
                                    print(msg_success)
                            
                            except Exception as _cma_run_e:
                                print(f"[CMA-Refine][WARN] CMA运行失败: {_cma_run_e}")
            except Exception as _cma_e:
                if it == args.cma_refine_every:
                    print(f"[CMA-Refine][WARN] CMA微调失败: {_cma_e}")
        
        # 测试集实时验证逻辑
        if args.test_verify_every > 0 and it % args.test_verify_every == 0:
            try:
                cur_prog_test, cur_best_test = agent.get_best_program()
                if cur_prog_test:
                    # 准备测试集轨迹
                    test_traj_names = []
                    if args.test_traj_preset:
                        preset = args.test_traj_preset
                        if preset == 'test_challenge':
                            test_traj_names = ['zigzag3d','lemniscate3d','random_waypoints','spiral_in_out','stairs','coupled_surface']
                    
                    if test_traj_names:
                        # 构建测试集的干扰和奖励配置
                        test_disturbances = build_disturbances(args.test_disturbance)
                        test_reward_weights, test_reward_ks = get_reward_profile(args.reward_profile)
                        
                        # 在测试集上评估
                        test_rewards = []
                        for tj_name in test_traj_names:
                            _tj_actual = tj_name if tj_name != 'random_wp' else 'random_waypoints'
                            _traj = build_trajectory(_tj_actual)
                            
                            # 创建控制器
                            _ctrl_test = PiLightSegmentedPIDController(
                                drone_model=DroneModel.CF2X,
                                program=cur_prog_test,
                                compose_by_gain=args.compose_by_gain,
                                clip_P=None,
                                clip_I=None,
                                clip_D=args.test_clip_D,
                                semantics='compose_by_gain' if args.compose_by_gain else 'first_match'
                            )
                            
                            # 创建 SimulationTester
                            _tst = SimulationTester(
                                controller=_ctrl_test,
                                test_scenarios=test_disturbances,
                                weights=test_reward_weights,
                                trajectory=_traj,
                                duration_sec=args.test_duration,
                                log_skip=args.log_skip,
                                output_folder='01_pi_flight/results/test_verify',
                                gui=False,
                                in_memory=True,
                                quiet=True
                            )
                            
                            try:
                                _rwd_test = _tst.run()
                                test_rewards.append(float(_rwd_test))
                            except Exception:
                                test_rewards.append(-1e18)
                        
                        # 聚合测试集得分
                        if test_rewards:
                            if args.test_aggregate == 'harmonic':
                                valid_test_rws = [r for r in test_rewards if r > 0]
                                if valid_test_rws:
                                    test_score = len(valid_test_rws) / sum(1.0/r for r in valid_test_rws)
                                else:
                                    test_score = -1e18
                            elif args.test_aggregate == 'min':
                                test_score = min(test_rewards)
                            else:
                                test_score = sum(test_rewards) / len(test_rewards)
                            
                            # 检查是否超过历史最佳
                            if not hasattr(agent, '_best_test_score'):
                                agent._best_test_score = -1e18  # type: ignore
                                agent._best_test_program = None  # type: ignore
                            
                            if test_score > agent._best_test_score + 1e-9:  # type: ignore
                                agent._best_test_score = test_score  # type: ignore
                                agent._best_test_program = cur_prog_test  # type: ignore
                                
                                # 立即保存到主文件
                                test_meta = {
                                    'best_score': float(cur_best_test),
                                    'test_score': float(test_score),
                                    'test_verified_at_iter': int(it),
                                    'iters': int(it),
                                    'trajectories': traj_names,
                                    'test_trajectories': test_traj_names,
                                    'aggregate': args.aggregate,
                                    'test_aggregate': args.test_aggregate,
                                    'disturbance': args.disturbance,
                                    'test_disturbance': args.test_disturbance,
                                }
                                save_program_json(cur_prog_test, args.save_program, meta=test_meta)
                                
                                msg = f"[TestVerify] @ iter {it}: 测试集得分 {test_score:.4f} > 历史最佳 {agent._best_test_score - test_score + test_score:.4f}，已保存"  # type: ignore
                                if use_tqdm:
                                    try:
                                        from tqdm import tqdm
                                        tqdm.write(msg)
                                    except Exception:
                                        print(msg)
                                else:
                                    print(msg)
                            else:
                                msg = f"[TestVerify] @ iter {it}: 测试集得分 {test_score:.4f} <= 历史最佳 {agent._best_test_score:.4f}，跳过保存"  # type: ignore
                                if use_tqdm:
                                    try:
                                        from tqdm import tqdm
                                        tqdm.write(msg)
                                    except Exception:
                                        print(msg)
                                else:
                                    print(msg)
            except Exception as _test_e:
                if it == args.test_verify_every:
                    print(f"[TestVerify][WARN] 测试集验证失败: {_test_e}")

        if args.save_every and it % args.save_every == 0:
            best_prog_mid, best_score_mid = agent.get_best_program()
            try:
                cur_train_for_ckpt = float(short_eval_all(best_prog_mid)) if best_prog_mid else float('nan')
            except Exception:
                cur_train_for_ckpt = float('nan')
            if not isinstance(cur_train_for_ckpt, float) or (cur_train_for_ckpt != cur_train_for_ckpt):
                cur_train_for_ckpt = float(best_score_mid)
            if cur_train_for_ckpt <= ckpt_last_saved_train + 1e-9:
                skip_msg = f"[保存] 跳过 checkpoint @ {it}（训练分未提升：{cur_train_for_ckpt:.6f} <= {ckpt_last_saved_train:.6f}）"
                if use_tqdm:
                    try:
                        from tqdm import tqdm
                        tqdm.write(skip_msg)
                    except Exception:
                        print(skip_msg)
                else:
                    print(skip_msg)
            else:
                ckpt_last_saved_train = cur_train_for_ckpt
            meta_mid = {
                'best_score': float(best_score_mid),
                'iters': it,
                'trajectories': traj_names,
                'aggregate': args.aggregate,
                'disturbance': args.disturbance,
                'partial': True,
                'best_score_short': float(cur_train_for_ckpt)
            }
            try:
                import os as _os
                _ck_dir = _os.path.join(_os.path.dirname(args.save_program) or '.', 'checkpoints')
                _os.makedirs(_ck_dir, exist_ok=True)
                _ck_path = _os.path.join(_ck_dir, f"best_program_iter_{it:06d}.json")
            except Exception:
                _ck_path = args.save_program + f".iter_{it:06d}.json"
            save_program_json(best_prog_mid, _ck_path, meta=meta_mid)
            save_search_history(agent.best_history, args.save_history)
            msg = f"[保存] 中间检查点 @ {it} -> {_ck_path} (short={cur_train_for_ckpt:.6f})"
            if use_tqdm:
                try:
                    from tqdm import tqdm
                    tqdm.write(msg)
                except Exception:
                    print(msg)
            else:
                print(msg)

        best_prog, best_reward_now = agent.get_best_program()
        prog_size = len(best_prog) if best_prog else 0
    try:
        cur_hash_for_log = hash_program(best_prog) if best_prog else None
        if best_prog and cur_hash_for_log != log_best_train_hash:
            log_best_train_score = float(short_eval_all(best_prog))
            log_best_train_hash = cur_hash_for_log
    except Exception:
        log_best_train_score = float(best_reward_now)

    # 结构爆发逻辑已移除，保持主流程简洁

    elapsed_total = time.time() - t0
    it_per_sec = it/elapsed_total if elapsed_total > 0 else 0.0
    try:
        with open(iter_log_path, 'a', encoding='utf-8') as f:
            try:
                rdec = int(getattr(args, 'rebound_decay_iters', 0) or 0)
                rtarget = float(getattr(args, 'rebound_target_eps', 0.12) or 0.12)
                if rdec > 0 and getattr(agent, '_rebound_until_iter', 0) and getattr(agent, 'total_iterations_done', 0) < getattr(agent, '_rebound_until_iter', 0):
                    remaining = int(getattr(agent, '_rebound_until_iter', 0)) - int(getattr(agent, 'total_iterations_done', 0))
                    ratio = min(1.0, max(0.0, 1.0 - (remaining / max(1, rdec))))
                    cur_min = float(getattr(agent, '_epsilon_rebound', 0.0))
                    decayed = max(rtarget, cur_min * (1.0 - ratio) + rtarget * ratio)
                    if hasattr(agent, 'epsilon'):
                        agent.epsilon = max(rtarget, min(1.0, float(getattr(agent, 'epsilon', 0.0))))
                    setattr(agent, '_epsilon_rebound', decayed)
            except Exception:
                pass
            eps = getattr(agent, 'epsilon', 0.0)
            rebound_active = 1 if getattr(agent, 'total_iterations_done', 0) < getattr(agent, '_rebound_until_iter', 0) else 0
            bcur_logged = float(log_best_train_score) if best_prog else float('nan')
            f.write(f"{it},{bcur_logged:.6f},{prog_size},{elapsed_total:.2f},{it_per_sec:.3f},{eps:.4f},{rebound_active}\n")
    except Exception as _log_e:
        if it == 1:
            print(f"[IterLog][WARN] 写入失败: {_log_e}")

    try:
        if pool_holder.get('pool') is not None:
            pool_holder['pool'].close()  # type: ignore
            pool_holder['pool'].join()   # type: ignore
    except Exception as _pc:
        print(f"[Parallel][WARN] 进程池关闭失败: {_pc}")

    # 结束：保存搜索历史与当前训练集最优程序
    save_search_history(agent.best_history, args.save_history)
    try:
        final_prog, final_score = agent.get_best_program()
        
        # 验证最终程序性能，如果比 warm_start 原程序差，保留原程序
        warm_start_prog = None
        warm_start_score = None
        if args.warm_start_cmaes and args.warm_start_path:
            try:
                with open(args.warm_start_path, 'r', encoding='utf-8') as _wsf:
                    ws_data = json.load(_wsf)
                    if 'rules' in ws_data and ws_data['rules']:
                        warm_start_prog = deserialize_program(ws_data)
                        # 获取原验证分数
                        warm_start_score = ws_data.get('meta', {}).get('verified_score', None)
                        if warm_start_score is None:
                            # 如果没有验证分数，快速评估一次
                            print("[PostValidate] Warm-start 程序无 verified_score，快速评估...")
                            warm_start_score = float(short_eval_all(warm_start_prog))
            except Exception as _ws_load_e:
                print(f"[PostValidate][WARN] 无法载入 warm_start 程序: {_ws_load_e}")
        
        # 快速评估最终程序
        final_validated_score = float(short_eval_all(final_prog))
        print(f"[PostValidate] 训练后程序得分: {final_validated_score:.6f}")
        
        prog_to_save = final_prog
        score_to_save = final_validated_score
        
        if warm_start_prog is not None and warm_start_score is not None:
            print(f"[PostValidate] Warm-start 原程序得分: {warm_start_score:.6f}")
            if final_validated_score < warm_start_score * 0.98:  # 如果下降超过2%
                print(f"[PostValidate][WARN] 训练后性能下降 {((warm_start_score - final_validated_score)/warm_start_score*100):.1f}%，保留原程序")
                prog_to_save = warm_start_prog
                score_to_save = warm_start_score
            else:
                print(f"[PostValidate][OK] 训练后性能提升或持平，保存新程序")
        
        meta = {
            'best_score': float(final_score if isinstance(final_score,(int,float)) else float('nan')),
            'validated_score': float(score_to_save),
            'iters': int(getattr(agent, 'total_iterations_done', 0)),
            'trajectories': traj_names,
            'aggregate': args.aggregate,
            'disturbance': args.disturbance,
        }
        save_program_json(prog_to_save, args.save_program, meta=meta)
        print(f"[Summary] 训练完成，best_train={meta['best_score']:.6f}, validated={meta['validated_score']:.6f}\nSaved program => {args.save_program}\nSaved history => {args.save_history}")
    except Exception as _sum_e:
        print(f"[Summary][WARN] 保存最终程序失败（已保存历史）: {_sum_e}")

if __name__ == '__main__':
    main()
