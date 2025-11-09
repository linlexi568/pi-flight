r"""
Verify a saved best_program.json under a standardized evaluation pipeline
and write back verified_score and verified_settings to eliminate metric drift.

Usage (PowerShell example):
    & .\.venv\Scripts\python.exe utilities\verify_program.py `
        --program 01_pi_flight\results\best_program.json `
        --traj_preset test_challenge `
        --aggregate harmonic `
        --disturbance mild_wind `
        --duration 20 `
        --log-skip 2 `
        --clip-D 1.2 `
        --compose-by-gain `
        --inplace
"""
from __future__ import annotations
import os, sys, argparse, json, importlib.util, time
from typing import List, Dict, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utilities.reward_profiles import get_reward_profile, describe_profile  # type: ignore

_PI_FLIGHT_DIR = os.path.join(_ROOT, '01_pi_flight')
_PI_LIGHT_DIR = os.path.join(_ROOT, '01_pi_light')
def _load_pilight_loader():
    """
    动态加载 serialization.load_program_json，并确保所用包名与控制器加载时一致：
    - 01_pi_flight 使用包别名 piflight_verify
    - 01_pi_light 使用包别名 pilight_verify（仅作回退兼容）
    这样，serialization.py 内的相对导入（.dsl 等）才能找到正确的父包。
    """
    # Prefer 01_pi_flight first, fallback to 01_pi_light
    for base, pkg_name in ((_PI_FLIGHT_DIR, 'piflight_verify'), (_PI_LIGHT_DIR, 'pilight_verify')):
        ser_file = os.path.join(base, 'serialization.py')
        if os.path.isfile(ser_file):
            full_name = f"{pkg_name}.serialization"
            spec = importlib.util.spec_from_file_location(full_name, ser_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # 确保父包已被 _load_pilight_controller 注册；若还没有，先临时占位
                parent_pkg = sys.modules.get(pkg_name)
                if parent_pkg is None:
                    pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader(pkg_name, loader=None))  # type: ignore
                    pkg.__path__ = [base]  # type: ignore[attr-defined]
                    sys.modules[pkg_name] = pkg
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)  # type: ignore
                load_program_json = getattr(mod, 'load_program_json', None)
                if load_program_json:
                    return load_program_json
    raise ImportError('Failed to load serialization.load_program_json from 01_pi_flight or 01_pi_light')

def _load_pilight_controller():
    # Prefer 01_pi_flight first, fallback to 01_pi_light
    for base, name in ((_PI_FLIGHT_DIR, 'piflight_verify'), (_PI_LIGHT_DIR, 'pilight_verify')):
        init_file = os.path.join(base, '__init__.py')
        if os.path.isfile(init_file):
            spec = importlib.util.spec_from_file_location(name, init_file, submodule_search_locations=[base])
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)  # type: ignore
                PiLightSegmentedPIDController = getattr(mod, 'PiLightSegmentedPIDController', None)
                if PiLightSegmentedPIDController is not None:
                    return PiLightSegmentedPIDController
    raise ImportError('Failed to load PiLightSegmentedPIDController from 01_pi_flight or 01_pi_light')

def build_trajectory(name: str) -> Dict[str, Any]:
    # Mirror main_no_gui/train builders for consistency
    if name == 'figure8':
        return { 'type': 'figure_8','initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8,'B': 0.5,'period': 12}}
    if name == 'helix':
        return { 'type': 'helix','initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7,'period': 10,'v_z': 0.15}}
    if name == 'circle':
        return { 'type': 'circle','initial_xyz': [0, 0, 0.8], 'params': {'R': 0.9,'period': 10}}
    if name == 'square':
        return { 'type': 'square','initial_xyz': [0, 0, 0.8], 'params': {'side_len': 1.2,'period': 12,'corner_hold': 0.5}}
    if name == 'step_hover':
        return { 'type': 'step_hover','initial_xyz': [0, 0, 0.6], 'params': {'z2': 1.2,'switch_time': 6.0}}
    if name == 'spiral_out':
        return { 'type': 'spiral_out','initial_xyz': [0, 0, 0.6], 'params': {'R0': 0.2,'k': 0.05,'period': 9,'v_z':0.02}}
    if name == 'zigzag3d':
        return { 'type': 'zigzag3d','initial_xyz':[0,0,0.7], 'params': {'amplitude':0.8,'segments':6,'z_inc':0.08,'period':14.0}}
    if name == 'lemniscate3d':
        return { 'type': 'lemniscate3d','initial_xyz':[0,0,0.7], 'params': {'a':0.9,'period':16.0,'z_amp':0.25}}
    if name in ('random_wp','random_waypoints'):
        return { 'type': 'random_waypoints','initial_xyz':[0,0,0.8], 'params': {'hold_time':1.2, 'transition':'linear'}}
    if name == 'spiral_in_out':
        return { 'type': 'spiral_in_out','initial_xyz':[0,0,0.7], 'params': {'R_in':0.9,'R_out':0.2,'period':14,'z_wave':0.15}}
    if name == 'stairs':
        return { 'type': 'stairs','initial_xyz':[0,0,0.6], 'params': {'levels':[0.6,0.9,1.2], 'segment_time':3.0}}
    if name == 'coupled_surface':
        return { 'type': 'coupled_surface','initial_xyz':[0,0,0.8], 'params': {'ax':0.9,'ay':0.7,'f1':1.0,'f2':2.0,'phase':1.0472,'z_amp':0.25,'surf_amp':0.15}}
    # 测试集极端版本 (训练/测试分离)
    if name == 'coupled_surface_extreme':
        return { 'type': 'coupled_surface','initial_xyz': [0, 0, 0.9], 'params': {'ax': 1.1,'ay': 0.9,'f1': 1.5,'f2': 3.0,'phase': 0.7,'z_base': 0.9,'z_amp': 0.35,'surf_amp': 0.22}}
    if name == 'zigzag3d_aggressive':
        return { 'type': 'zigzag3d','initial_xyz': [0, 0, 0.6], 'params': {'amplitude': 1.1,'segments': 8,'z_inc': 0.12,'period': 10.0}}
    if name == 'lemniscate3d_wild':
        return { 'type': 'lemniscate3d','initial_xyz': [0, 0, 0.6], 'params': {'a': 1.2,'period': 12.0,'z_amp': 0.40}}
    if name == 'spiral_chaotic':
        return { 'type': 'spiral_in_out','initial_xyz': [0, 0, 0.65], 'params': {'R_in': 1.1,'R_out': 0.15,'period': 10.0,'z_wave': 0.25}}
    if name == 'stairs_harsh':
        return { 'type': 'stairs','initial_xyz': [0, 0, 0.5], 'params': {'levels': [0.5, 0.8, 1.1, 1.4],'segment_time': 2.2}}
    raise ValueError(f"Unknown trajectory: {name}")

def build_preset(preset: str) -> List[str]:
    if preset == 'train_core':
        return ['figure8','helix','circle','square','step_hover','spiral_out']
    if preset in ('test_challenge','pi_strong_train','pi_strong_test'):
        return ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface']
    if preset == 'test_extreme':
        return ['coupled_surface_extreme','zigzag3d_aggressive','lemniscate3d_wild','spiral_chaotic','stairs_harsh']
    if preset == 'full_eval':
        return ['figure8','helix','circle','square','step_hover','spiral_out',
                'zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface']
    raise ValueError(f"Unknown preset: {preset}")

def build_disturbances(preset: str | None):
    if not preset:
        return []
    if preset == 'mild_wind':
        return [
            {'type': 'SUSTAINED_WIND','info':'mild','start_time':3.0,'end_time':6.0,'force':[0.004,0.0,0.0]},
            {'type': 'PULSE','time':8.0,'force':[0.008,-0.004,0.0],'info':'pulse'}
        ]
    if preset == 'stress':
        return [
            {'type': 'SUSTAINED_WIND','info':'stress:steady_wind','start_time':2.5,'end_time':6.5,'force':[0.007,0.0,0.0]},
            {'type': 'GUSTY_WIND','info':'stress:gusty_wind','start_time':7.5,'end_time':11.5,'base_force':[0.0,-0.004,0.0],'gust_frequency':6.0,'gust_amplitude':0.006},
            {'type': 'MASS_CHANGE','info':'stress:mass_up','time':12.0,'mass_multiplier':1.08},
            {'type': 'PULSE','info':'stress:pulse','time':14.0,'force':[-0.008,0.008,0.0]}
        ]
    raise ValueError(f"Unknown disturbance preset: {preset}")

def harmonic_mean(values: List[float]) -> float:
    return len(values)/sum(1.0/(v+1e-9) for v in values)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--program', type=str, default=os.path.join('01_pi_flight','results','best_program.json'))
    ap.add_argument('--traj_list', type=str, nargs='*', default=None)
    ap.add_argument('--traj_preset', type=str, default='test_challenge',
                    choices=['train_core','test_challenge','full_eval','pi_strong_train','pi_strong_test','test_extreme',
                             'eth_rpg_core','eth_rpg_challenge','eth_rpg_extreme','academic_full'])
    ap.add_argument('--aggregate', type=str, default='harmonic', choices=['mean','min','harmonic'])
    ap.add_argument('--disturbance', type=str, default='mild_wind', choices=[None,'mild_wind','stress'])
    ap.add_argument('--duration', type=int, default=20)
    ap.add_argument('--log-skip', type=int, default=2)
    ap.add_argument('--reward_profile', type=str, default='pilight_boost', choices=['default','pilight_boost','pilight_freq_boost','control_law_discovery'])
    ap.add_argument('--compose-by-gain', action='store_true')
    ap.add_argument('--clip-P', type=float, default=None)
    ap.add_argument('--clip-I', type=float, default=None)
    ap.add_argument('--clip-D', type=float, default=1.2)
    ap.add_argument('--inplace', action='store_true', help='写回 verified_score 与 verified_settings 到 program JSON')
    return ap.parse_args()

def main():
    args = parse_args()
    # Print profile for visibility
    print(describe_profile(args.reward_profile))
    weights, ks = get_reward_profile(args.reward_profile)

    # Prefer academic benchmarks if requested
    use_academic = False
    traj_names = []
    trajectories = []
    disturbances = []
    try:
        from utilities.academic_benchmarks import build_preset as ab_build_preset, \
            build_trajectory as ab_build_traj, build_disturbances as ab_build_dist, \
            ACADEMIC_PRESET_NAMES
        if args.traj_list:
            # If user gave explicit names, try academic first then fallback
            for n in args.traj_list:
                try:
                    trajectories.append(ab_build_traj(n))
                    traj_names.append(n)
                    use_academic = True
                except Exception:
                    trajectories.append(build_trajectory(n))
                    traj_names.append(n)
        else:
            if args.traj_preset in ACADEMIC_PRESET_NAMES:
                traj_names = ab_build_preset(args.traj_preset)
                trajectories = [ab_build_traj(n) for n in traj_names]
                use_academic = True
            else:
                traj_names = build_preset(args.traj_preset)
                trajectories = [build_trajectory(n) for n in traj_names]
        # disturbances: if academic used, still reuse our standardized definitions for parity
        disturbances = ab_build_dist(args.disturbance) if use_academic else build_disturbances(args.disturbance)
    except Exception:
        # Fallback entirely to legacy builders
        traj_names = args.traj_list if args.traj_list else build_preset(args.traj_preset)
        trajectories = [build_trajectory(n) for n in traj_names]
        disturbances = build_disturbances(args.disturbance)

    from utilities.isaac_tester import SimulationTester  # type: ignore

    # 必须先加载包 (__init__) 注册 pilight_verify，再加载 serialization 子模块
    PiLightSegmentedPIDController = _load_pilight_controller()
    load_program_json = _load_pilight_loader()

    if not os.path.isfile(args.program):
        raise FileNotFoundError(f"Program JSON not found: {args.program}")
    program = load_program_json(args.program)
    print(f"[加载] 程序: {args.program} (规则数={len(program)})")

    scores: List[float] = []
    per_traj: List[Dict[str, Any]] = []
    for t in trajectories:
        ctrl = PiLightSegmentedPIDController(drone_model="cf2x", program=program,
                                             compose_by_gain=bool(args.compose_by_gain),
                                             clip_P=args.clip_P, clip_I=args.clip_I, clip_D=args.clip_D)
        tester = SimulationTester(ctrl, disturbances, weights, duration_sec=args.duration,
                                  output_folder=os.path.join('01_pi_flight','results','mcts_eval'), gui=False,
                                  trajectory=t, log_skip=max(1,int(args.log_skip)), in_memory=True, quiet=True)
        r = tester.run()
        scores.append(float(r))
        per_traj.append({'traj': t['type'], 'reward': float(r)})

    if args.aggregate == 'mean':
        agg = float(sum(scores)/len(scores))
    elif args.aggregate == 'min':
        agg = float(min(scores))
    else:
        agg = float(harmonic_mean(scores))

    print("\n[Verified] 聚合得分:", f"{agg:.6f}")
    print("逐轨迹: ", per_traj)

    if args.inplace:
        try:
            with open(args.program, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取 program JSON 失败，放弃写回: {e}")
            return
        settings = {
            'traj_names': traj_names,
            'aggregate': args.aggregate,
            'disturbance': args.disturbance,
            'duration': args.duration,
            'log_skip': args.log_skip,
            'reward_profile': args.reward_profile,
            'compose_by_gain': bool(args.compose_by_gain),
            'clip_P': args.clip_P, 'clip_I': args.clip_I, 'clip_D': args.clip_D,
        }
        data.setdefault('meta', {})['verified_score'] = float(agg)
        data['meta']['verified_settings'] = settings
        data['meta']['verified_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        # 保留 per-traj 结果便于诊断
        data['meta']['verified_per_traj'] = per_traj
        with open(args.program, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[写回] 已更新 verified_score 至 {args.program}")

if __name__ == '__main__':
    main()
