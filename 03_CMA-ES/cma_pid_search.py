import json
import time
import argparse
from pathlib import Path
import numpy as np
import sys
import random

# 保证项目根目录在路径中
CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import cma
except ImportError:
    raise SystemExit("[错误] 未安装 cma 库，请先: pip install cma")

from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from test import SimulationTester  # type: ignore
from utilities.reward_profiles import get_reward_profile, describe_profile

_PRESET_MAP = {
    'train_core': ['figure8','helix','circle','square','step_hover','spiral_out'],
    'test_challenge': ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
    'full_eval': ['figure8','helix','circle','square','step_hover','spiral_out',
                  'zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
    # 强化 PI-Light 优势的训练/测试预设（突出非平稳/多相位），两者当前相同但语义区分
    'pi_strong_train': ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
    'pi_strong_test':  ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
}


def build_trajectory(name: str, duration_sec: int, traj_seed: int) -> dict:
    if name == 'figure8':
        return {'type': 'figure_8','initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8,'B': 0.5,'period': 12}}
    if name == 'helix':
        return {'type': 'helix','initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7,'period': 10,'v_z': 0.15}}
    if name == 'circle':
        return {'type': 'circle','initial_xyz': [0, 0, 0.8], 'params': {'R': 0.9,'period': 10}}
    if name == 'square':
        return {'type': 'square','initial_xyz': [0, 0, 0.8], 'params': {'side_len': 1.2,'period': 12,'corner_hold': 0.5}}
    if name == 'step_hover':
        return {'type': 'step_hover','initial_xyz': [0, 0, 0.6], 'params': {'z2': 1.2,'switch_time': duration_sec/2.0}}
    if name == 'spiral_out':
        return {'type': 'spiral_out','initial_xyz': [0, 0, 0.6], 'params': {'R0': 0.2,'k': 0.05,'period': 9,'v_z':0.02}}
    if name == 'zigzag3d':
        return {'type': 'zigzag3d','initial_xyz':[0,0,0.6], 'params': {'amplitude':0.8,'segments':6,'z_inc':0.08,'period':14}}
    if name == 'lemniscate3d':
        return {'type': 'lemniscate3d','initial_xyz':[0,0,0.8], 'params': {'a':0.9,'period':16,'z_amp':0.25}}
    if name in ('random_wp','random_waypoints'):
        rng = random.Random(traj_seed)
        waypoints = [[rng.uniform(-0.9,0.9), rng.uniform(-0.9,0.9), rng.uniform(0.6,1.2)] for _ in range(6)]
        return {'type': 'random_waypoints','initial_xyz':[0,0,0.7], 'params': {'waypoints': waypoints, 'hold_time':1.2, 'transition':'spline'}}
    if name == 'spiral_in_out':
        return {'type': 'spiral_in_out','initial_xyz':[0,0,0.7], 'params': {'R_in':0.9,'R_out':0.2,'period':14,'z_wave':0.15}}
    if name == 'stairs':
        return {'type': 'stairs','initial_xyz':[0,0,0.5], 'params': {'levels':[0.5,0.7,0.9,1.1,0.8], 'segment_time':3.0}}
    if name == 'coupled_surface':
        return {'type': 'coupled_surface','initial_xyz':[0,0,0.8], 'params': {'ax':0.9,'ay':0.7,'f1':1.0,'f2':2.0,'phase':1.047,'z_amp':0.25,'surf_amp':0.15}}
    raise ValueError(f"未知轨迹: {name}")


def build_disturbances(preset: str | None) -> list:
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
    raise ValueError(f"未知扰动预设: {preset}")


def aggregate_scores(scores: list[float], mode: str) -> float:
    if not scores:
        return float('nan')
    if mode == 'mean':
        return float(sum(scores)/len(scores))
    if mode == 'min':
        return float(min(scores))
    if mode == 'harmonic':
        import math
        return len(scores)/sum(1/(s+1e-9) for s in scores)
    return float(sum(scores)/len(scores))


def resolve_traj_names(manual_list, preset: str | None, fallback: str | None) -> list[str]:
    if manual_list:
        return list(manual_list)
    if preset:
        names = _PRESET_MAP.get(preset, [])
        if not names:
            raise ValueError(f"未知轨迹预设: {preset}")
        return list(dict.fromkeys(names))  # 保序去重
    if fallback:
        return [fallback]
    raise ValueError('必须至少指定一条轨迹（通过 --traj_list / --traj_preset / --trajectory）')


def evaluate_pid(params,
                 duration_sec: int,
                 trajectories: list[dict],
                 disturbances: list[dict],
                 reward_weights: dict,
                 aggregate: str,
                 seed: int = 0,
                 output_dir: str = '03_CMA-ES/results/logs') -> tuple[float, list[dict]]:
    np.random.seed(seed)
    per_traj = []
    scores = []
    for traj in trajectories:
        controller = DSLPIDControl(drone_model=DroneModel("cf2x"))
        p_gain, i_gain, d_gain = params
        controller.P_COEFF_TOR[:] = p_gain * controller.P_COEFF_TOR
        controller.I_COEFF_TOR[:] = i_gain * controller.I_COEFF_TOR
        controller.D_COEFF_TOR[:] = d_gain * controller.D_COEFF_TOR
        tester = SimulationTester(controller=controller,
                                  test_scenarios=disturbances,
                                  weights=reward_weights,
                                  duration_sec=duration_sec,
                                  output_folder=output_dir,
                                  gui=False,
                                  trajectory=traj)
        reward = tester.run()
        scores.append(reward)
        per_traj.append({'traj': traj.get('type', 'unknown'), 'reward': float(reward)})
    agg = aggregate_scores(scores, aggregate)
    return float(agg), per_traj


def main():
    parser = argparse.ArgumentParser(description='CMA-ES 搜索 (P,I,D) 缩放系数')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--popsize', type=int, default=10)
    parser.add_argument('--duration', type=int, default=10, help='单次仿真持续时间 (秒)')
    parser.add_argument('--trajectory', type=str, default=None,
                        choices=['figure8','helix','circle','square','step_hover','spiral_out',
                                 'zigzag3d','lemniscate3d','random_wp','random_waypoints','spiral_in_out','stairs','coupled_surface'],
                        help='单一轨迹（与 --traj_list 互斥；若指定 --traj_preset 会忽略此项）')
    parser.add_argument('--traj_list', type=str, nargs='*', default=None,
                        help='多轨迹列表，例如: figure8 helix circle square step_hover spiral_out ...')
    parser.add_argument('--traj_preset', type=str, default='train_core',
                        choices=list(_PRESET_MAP.keys()),
                        help='训练用轨迹预设（默认 train_core，覆盖 6 条训练轨迹）')
    parser.add_argument('--eval_traj_preset', type=str, default=None,
                        choices=list(_PRESET_MAP.keys()),
                        help='可选：搜索完成后在指定预设（如 full_eval）上复评最佳参数')
    parser.add_argument('--traj_seed', type=int, default=42, help='随机轨迹（如 random_wp）生成种子，确保可复现')
    parser.add_argument('--aggregate', type=str, default='harmonic', choices=['mean','min','harmonic'], help='多轨迹聚合方式')
    parser.add_argument('--disturbance', type=str, default='none', choices=['none','mild_wind','stress'], help='扰动预设（none=无扰动）')
    parser.add_argument('--reward_profile', type=str, default='pilight_boost', choices=['default','pilight_boost'], help='奖励权重配置')
    parser.add_argument('--seed', type=int, default=0)
    # 移除 GUI 相关：脚本强制 headless
    # 默认改为输出 best_program 风格，便于统一评测与热启动
    parser.add_argument('--output', type=str, default='03_CMA-ES/results/best_program.json')
    parser.add_argument('--report-every', type=int, default=1, help='每隔多少代打印进度 (默认每代)')
    parser.add_argument('--use-tqdm', action='store_true', help='使用 tqdm 进度条显示代数进展')
    parser.add_argument('--improve-patience', type=int, default=0, help='若指定且连续若干代无提升则提前停止 (0=禁用)')
    # 兼容开关：如需旧格式（包含 best_params 等调参历史），可显式要求额外写一份
    parser.add_argument('--emit-legacy-json', action='store_true', help='同时输出旧格式 JSON（含 best_params/history）')
    parser.add_argument('--legacy-output', type=str, default='03_CMA-ES/results/cma_es_pid_result.json', help='旧格式 JSON 输出路径')
    args = parser.parse_args()

    traj_names = resolve_traj_names(args.traj_list, args.traj_preset, args.trajectory or 'figure8')
    if args.traj_preset:
        print(f"[INFO] 训练轨迹预设 {args.traj_preset}: {traj_names}")
    eval_traj_names = []
    if args.eval_traj_preset:
        eval_traj_names = list(dict.fromkeys(_PRESET_MAP.get(args.eval_traj_preset, [])))
        print(f"[INFO] 额外评测轨迹预设 {args.eval_traj_preset}: {eval_traj_names}")
    disturbances = build_disturbances(None if args.disturbance == 'none' else args.disturbance)
    if disturbances:
        print(f"[INFO] 启用扰动预设: {args.disturbance}")
    trajectories = [build_trajectory(name, args.duration, args.traj_seed) for name in traj_names]
    eval_trajectories = [build_trajectory(name, args.duration, args.traj_seed) for name in eval_traj_names] if eval_traj_names else []
    reward_weights, _reward_ks = get_reward_profile(args.reward_profile)
    print(describe_profile(args.reward_profile))
    print(f"[INFO] 聚合方式: {args.aggregate}\n")

    x0 = [1.0, 1.0, 1.0]
    sigma0 = 0.3
    bounds = [[0.3, 0.0, 0.0], [3.0, 1.5, 2.0]]

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds': bounds,
        'popsize': args.popsize,
        'maxiter': args.iters,
        'seed': args.seed
    })

    history = []
    t_start = time.time()
    gen = 0
    best_so_far = -float('inf')
    best_per_traj = None
    no_improve_iters = 0

    tqdm_bar = None
    if args.use_tqdm:
        try:
            from tqdm import trange
            tqdm_bar = trange(args.iters, desc='CMA-ES', unit='gen')
            iter_source = tqdm_bar
        except Exception:
            print('[WARN] tqdm 未安装或不可用，回退到常规日志。')
            args.use_tqdm = False
            iter_source = range(args.iters)
    else:
        iter_source = range(args.iters)

    for _ in iter_source:
        solutions = es.ask()
        rewards = []  # 注意: rewards 里放的是 -reward (供最小化)
        gen_best = -float('inf')
        gen_best_params = None
        gen_best_per_traj = None
        for s in solutions:
            reward, per_traj = evaluate_pid(
                s,
                duration_sec=args.duration,
                trajectories=trajectories,
                disturbances=disturbances,
                reward_weights=reward_weights,
                aggregate=args.aggregate,
                seed=args.seed,
                output_dir='03_CMA-ES/results/logs'
            )
            rewards.append(-reward)
            history.append({'gen': gen, 'params': list(map(float, s)), 'reward': float(reward), 'per_traj': per_traj})
            if reward > gen_best:
                gen_best = reward
                gen_best_params = s
                gen_best_per_traj = per_traj
        es.tell(solutions, rewards)
        gen += 1
        improved = False
        if gen_best > best_so_far:
            best_so_far = gen_best
            best_per_traj = gen_best_per_traj
            improved = True
            no_improve_iters = 0
        else:
            no_improve_iters += 1
        # 仅当使用 tqdm 时更新进度条附加信息
        if tqdm_bar is not None:
            tqdm_bar.set_postfix(best=f"{best_so_far:.4f}", gen_best=f"{gen_best:.4f}")
        elif args.report_every > 0 and gen % args.report_every == 0:
            elapsed = time.time() - t_start
            speed = gen / elapsed if elapsed > 0 else 0.0
            remaining = args.iters - gen
            eta = remaining / speed if speed > 0 else float('inf')
            if gen_best_params is not None:
                print(f"[进度] 代 {gen}/{args.iters} | 本代最佳={gen_best:.4f} | 全局最佳={best_so_far:.4f} | 速度={speed:.2f} gen/s | ETA={eta:.1f}s")

        if args.improve_patience > 0 and no_improve_iters >= args.improve_patience:
            print(f"[停止] 连续 {no_improve_iters} 代未提升，提前收敛退出。")
            break

        if es.stop():
            break
    best_params = es.result.xbest
    best_reward = -es.result.fbest
    elapsed = time.time() - t_start

    # 统一主输出为 best_program 风格 JSON
    from datetime import datetime
    bp_payload = {
        'rules': [
            {
                'condition': {
                    'type': 'Binary',
                    'op': '>',
                    'left': {'type': 'Terminal', 'value': 1},
                    'right': {'type': 'Terminal', 'value': 0}
                },
                'action': [
                    {'type': 'Binary', 'op': 'set', 'left': {'type': 'Terminal', 'value': 'P'}, 'right': {'type': 'Terminal', 'value': float(best_params[0])}},
                    {'type': 'Binary', 'op': 'set', 'left': {'type': 'Terminal', 'value': 'I'}, 'right': {'type': 'Terminal', 'value': float(best_params[1])}},
                    {'type': 'Binary', 'op': 'set', 'left': {'type': 'Terminal', 'value': 'D'}, 'right': {'type': 'Terminal', 'value': float(best_params[2])}}
                ]
            }
        ],
        'meta': {
            'best_score': float(best_reward),
            'best_score_full': float(best_reward),  # 如需 eval 集合，可扩展为单独复评
            'best_score_short': float(best_reward),
            'iters': gen,
            'trajectories': traj_names,
            'aggregate': args.aggregate,
            'disturbance': None if args.disturbance == 'none' else args.disturbance,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    if eval_trajectories and best_params is not None:
        eval_reward, eval_per_traj = evaluate_pid(
            best_params,
            duration_sec=args.duration,
            trajectories=eval_trajectories,
            disturbances=disturbances,
            reward_weights=reward_weights,
            aggregate=args.aggregate,
            seed=args.seed,
            output_dir='03_CMA-ES/results/eval_full'
        )
        result['eval_reward'] = float(eval_reward)
        result['eval_per_traj'] = eval_per_traj
        print(f"[INFO] 额外评测 ({args.eval_traj_preset}) reward={eval_reward:.4f}")

    # 写主输出（best_program）
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(bp_payload, f, indent=2, ensure_ascii=False)
    print(f"[完成] CMA-ES 结果写入 {args.output} (best_program 格式) | reward={best_reward:.4f} | P,I,D={list(map(float,best_params))}")

    # 如需旧格式，额外再写一份
    if args.emit_legacy_json:
        legacy = {
            'best_params': list(map(float, best_params)),
            'best_reward': float(best_reward),
            'iterations': gen,
            'history': history,
            'elapsed_sec': elapsed,
            'config': {
                'iters': args.iters,
                'popsize': args.popsize,
                'duration': args.duration,
                'trajectories': traj_names,
                'aggregate': args.aggregate,
                'disturbance': None if args.disturbance == 'none' else args.disturbance,
                'reward_profile': args.reward_profile,
                'traj_seed': args.traj_seed,
                'seed': args.seed,
                'bounds': bounds
            },
            'best_per_traj': best_per_traj
        }
        Path(args.legacy_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.legacy_output, 'w', encoding='utf-8') as f:
            json.dump(legacy, f, indent=2, ensure_ascii=False)
        print(f"[完成] 兼容旧格式结果写入 {args.legacy_output}")

if __name__ == '__main__':
    main()
