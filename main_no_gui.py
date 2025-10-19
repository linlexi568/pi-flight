"""Non-GUI evaluation entry.

This script is a stripped version of main.py where all SimulationTester
invocations force gui=False (headless). The CLI omits --no_gui.
"""
import time, os, sys, numpy as np, argparse, json, importlib.util, warnings, io, contextlib
from utilities.reward_profiles import get_reward_profile, describe_profile
from utilities import paths

# --- 动态加载 PI-Flight (编号目录 01_pi_flight，兼容旧包名 pi_light) ---
PiLightSegmentedPIDController = None  # type: ignore
load_program_json = None  # type: ignore
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_PF_DIR = os.path.join(_ROOT_DIR, '01_pi_flight')
def _load_pilight():
    global PiLightSegmentedPIDController, load_program_json
    # Try new preferred package name first: pi_flight
    try:
        from pi_flight import PiLightSegmentedPIDController as _P, serialization as _S  # type: ignore
        PiLightSegmentedPIDController = _P
        load_program_json = getattr(_S, 'load_program_json', None)
        return
    except Exception:
        pass
    # Try legacy package name (pi_light)
    try:
        from pi_light import PiLightSegmentedPIDController as _P, serialization as _S  # type: ignore
        PiLightSegmentedPIDController = _P
        load_program_json = getattr(_S, 'load_program_json', None)
        return
    except Exception:
        pass
    # Fallback to numbered dir: try 01_pi_flight
    init_file_pf = os.path.join(_PF_DIR, '__init__.py')
    ser_file_pf = os.path.join(_PF_DIR, 'serialization.py')
    if os.path.isfile(init_file_pf):
        spec_pf = importlib.util.spec_from_file_location('piflight_dynamic', init_file_pf, submodule_search_locations=[_PF_DIR])
        if spec_pf and spec_pf.loader:
            module_pf = importlib.util.module_from_spec(spec_pf)
            sys.modules['piflight_dynamic'] = module_pf
            try:
                spec_pf.loader.exec_module(module_pf)  # type: ignore
                PiLightSegmentedPIDController = getattr(module_pf, 'PiLightSegmentedPIDController', None)
            except Exception as e:
                print(f"[Warn] 载入 01_pi_flight/__init__.py 失败: {e}")
    if load_program_json is None and os.path.isfile(ser_file_pf):
        spec2_pf = importlib.util.spec_from_file_location('piflight_dynamic.serialization', ser_file_pf)
        if spec2_pf and spec2_pf.loader:
            mod2_pf = importlib.util.module_from_spec(spec2_pf)
            sys.modules['piflight_dynamic.serialization'] = mod2_pf
            try:
                spec2_pf.loader.exec_module(mod2_pf)  # type: ignore
                load_program_json = getattr(mod2_pf, 'load_program_json', None)
            except Exception as e:
                print(f"[Warn] 载入 01_pi_flight/serialization 失败: {e}")
    # No longer attempt 01_pi_light folder (to allow safe deletion)
    missing = [n for n,v in {'PiLightSegmentedPIDController':PiLightSegmentedPIDController,'load_program_json':load_program_json}.items() if v is None]
    if missing:
        raise ImportError(f"无法导入 PI-Flight 组件: {missing} (尝试目录 01_pi_flight 或包 pi_flight/pi_light)")

_load_pilight()
from test import SimulationTester
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, default='compare_all', choices=['compare_all','gsn_only','baseline_only','pi_only','attn_only'], help='运行模式')
    ap.add_argument('--gsn_ckpt', type=str, default=os.path.join('04_nn_baselines','results','checkpoints','gsn_best.pt'), help='GSN 模型权重路径 (legacy: nn_baselines/results/...)')
    ap.add_argument('--attn_ckpt', type=str, default=os.path.join('04_nn_baselines','results','checkpoints','attn_best.pt'), help='Attention 模型权重路径')
    ap.add_argument('--duration_eval', type=int, default=20, help='评估阶段仿真时长(s)')
    ap.add_argument('--trajectory', type=str, default='figure8', choices=['figure8','helix','circle','square','step_hover','spiral_out','zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'], help='单一轨迹（与 --traj_list 互斥）')
    ap.add_argument('--traj_list', type=str, nargs='*', default=None, help='多轨迹列表: figure8 helix circle square step_hover spiral_out zigzag3d lemniscate3d random_wp spiral_in_out stairs coupled_surface ...')
    ap.add_argument('--traj_seed', type=int, default=42, help='随机轨迹（如 random_wp）生成种子，确保可重复性')
    ap.add_argument('--traj_preset', type=str, default=None,
                choices=['train_core', 'test_challenge', 'full_eval', 'pi_strong_train', 'pi_strong_test'],
                    help='轨迹预设集合：train_core=figure8 helix circle square step_hover spiral_out；'
                    'test_challenge=zigzag3d lemniscate3d random_wp spiral_in_out stairs coupled_surface；'
                    'full_eval=合并上述两组；'
                    'pi_strong_train=zigzag3d lemniscate3d random_wp spiral_in_out stairs coupled_surface；'
                    'pi_strong_test=zigzag3d lemniscate3d random_wp spiral_in_out stairs coupled_surface (同上，但保留语义区分)')
    ap.add_argument('--aggregate', type=str, default='mean', choices=['mean','min','harmonic'], help='多轨迹聚合方式')
    ap.add_argument('--test_traj_list', type=str, nargs='*', default=None, help='测试集轨迹列表（若提供则将输出 Train vs Test）')
    ap.add_argument('--test_traj_preset', type=str, default=None,
                    choices=['train_core', 'test_challenge', 'full_eval', 'pi_strong_train', 'pi_strong_test'],
                    help='测试集轨迹预设（语义同 --traj_preset）')
    ap.add_argument('--test_aggregate', type=str, default=None, choices=['mean','min','harmonic'], help='测试集聚合方式 (缺省沿用 --aggregate)')
    ap.add_argument('--run_cma', action='store_true', help='若指定且未找到默认 CMA-ES 结果文件，则内联运行一次轻量 CMA-ES 搜索')
    ap.add_argument('--cma_iters', type=int, default=15, help='内联 CMA-ES 最大迭代数')
    ap.add_argument('--cma_pop', type=int, default=8, help='内联 CMA-ES 种群大小')
    ap.add_argument('--save_summary', type=str, default=os.path.join(paths.SUMMARY_DIR,'summary_latest.json'), help='保存汇总 JSON 路径')
    ap.add_argument('--reward_profile', type=str, default='pilight_boost', choices=['default','pilight_boost'], help='奖励权重/系数配置档 (default|pilight_boost)')
    ap.add_argument('--deep-quiet', action='store_true', help='深度静音：抑制 pybullet build time 与 pkg_resources 弃用警告')
    ap.add_argument('--compose-by-gain', action='store_true', help='按 P/I/D 分量分别选最特异命中规则进行组合（替代首条命中即生效）')
    ap.add_argument('--semantics', type=str, default=None, choices=[None,'first_match','compose_by_gain','blend_topk'], help='规则组合语义（不指定则根据 --compose-by-gain 推断）')
    ap.add_argument('--require-k', type=int, default=0, help='当使用 blend_topk 时，要求至少命中 K 条规则，否则回退单一最特异（0=不要求）')
    ap.add_argument('--blend-topk-k', type=int, default=2, help='blend_topk 时每个增益分量融合的 top-k 规则数量（>=1）')
    ap.add_argument('--clip-P', type=float, default=None, help='将 P 增益裁剪至默认增益的最多 x 倍（>0 生效）')
    ap.add_argument('--clip-I', type=float, default=None, help='将 I 增益裁剪至默认增益的最多 x 倍（>0 生效）')
    ap.add_argument('--clip-D', type=float, default=None, help='将 D 增益裁剪至默认增益的最多 x 倍（>0 生效）')
    ap.add_argument('--gain-slew-limit', type=str, default=None, help='增益倍率每步最大变化量（标量或逗号分隔 P,I,D，例如 0.05 或 0.05,0.05,0.1）')
    ap.add_argument('--min-hold-steps', type=int, default=0, help='规则滞回保持步数（切换后至少保持 N 步不切换）')
    ap.add_argument('--disturbance', type=str, default=None, choices=[None,'mild_wind','stress'], help='评测扰动预设（与训练 build_disturbances 对齐）')
    ap.add_argument('--log-skip', type=int, default=1, help='日志降采样：每 N 个控制步记录一次（影响 jerk 等导数尺度；与训练对齐建议 2）')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if getattr(args, 'deep_quiet', False):
        # 抑制 pkg_resources 弃用警告
        warnings.filterwarnings("ignore", module="pkg_resources")
        # 捕获 pybullet build time 重复输出
        _silent = io.StringIO()
        with contextlib.redirect_stdout(_silent), contextlib.redirect_stderr(_silent):
            try:
                import pybullet  # noqa: F401
            except Exception:
                pass
    # Legacy rewrites -> numbered folders
    def _rewrite(pth: str, old_prefix: str, new_prefix: str, flag: str):
        if pth.startswith(old_prefix):
            tail = pth[len(old_prefix):].lstrip('/\\')
            new_path = os.path.join(new_prefix, tail)
            print(f"[LegacyPathWarning] {flag}: '{pth}' -> '{new_path}'")
            return new_path
        return pth
    args.gsn_ckpt = _rewrite(args.gsn_ckpt, 'nn_baselines', '04_nn_baselines', '--gsn_ckpt')

    # ---- 固定默认路径（去除命令行参数依赖） ----
    # Prefer new folder 01_pi_flight
    DEFAULT_PI_PROGRAM_CANDIDATES = [
        os.path.join('01_pi_flight','results','best_program.json'),
    ]
    DEFAULT_PI_PROGRAM = None
    for _p in DEFAULT_PI_PROGRAM_CANDIDATES:
        if os.path.isfile(_p):
            DEFAULT_PI_PROGRAM = _p
            break
    if DEFAULT_PI_PROGRAM is None:
        DEFAULT_PI_PROGRAM = DEFAULT_PI_PROGRAM_CANDIDATES[0]
    CMA_JSON_CANDIDATES = [
        os.path.join('03_CMA-ES','results','best_program.json'),
        os.path.join('03_CMA-ES','results','cma_es_pid_train.json'),
        os.path.join('03_CMA-ES','results','cma_es_pid_result.json'),
        os.path.join('03_CMA-ES','results','cma_es_pid_full_eval.json')
    ]
    def _pick_cma_json_path():
        for p in CMA_JSON_CANDIDATES:
            if os.path.isfile(p):
                return p, True
        # 若均不存在，返回主候选并标记未存在，用于内联搜索保存
        return CMA_JSON_CANDIDATES[0], False
    # ---- 构建轨迹集合（与 train_pi_light 对齐） ----
    traj_seed = getattr(args, 'traj_seed', 42)

    def build_trajectory(name: str):
        if name == 'figure8':
            return { 'type': 'figure_8','initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8,'B': 0.5,'period': 12}}
        if name == 'helix':
            return { 'type': 'helix','initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7,'period': 10,'v_z': 0.15}}
        if name == 'circle':
            return { 'type': 'circle','initial_xyz': [0, 0, 0.8], 'params': {'R': 0.9,'period': 10}}
        if name == 'square':
            return { 'type': 'square','initial_xyz': [0, 0, 0.8], 'params': {'side_len': 1.2,'period': 12,'corner_hold': 0.5}}
        if name == 'step_hover':
            # 训练期固定 6.0s 切换，避免随评测时长而变
            return { 'type': 'step_hover','initial_xyz': [0, 0, 0.6], 'params': {'z2': 1.2,'switch_time': 6.0}}
        if name == 'spiral_out':
            return { 'type': 'spiral_out','initial_xyz': [0, 0, 0.6], 'params': {'R0': 0.2,'k': 0.05,'period': 9,'v_z':0.02}}
        # ====== 复杂测试轨迹 ======
        if name == 'zigzag3d':
            # 与训练脚本对齐 initial 与参数
            return { 'type': 'zigzag3d','initial_xyz':[0,0,0.7], 'params': {'amplitude':0.8,'segments':6,'z_inc':0.08,'period':14.0}}
        if name == 'lemniscate3d':
            return { 'type': 'lemniscate3d','initial_xyz':[0,0,0.7], 'params': {'a':0.9,'period':16.0,'z_amp':0.25}}
        if name == 'random_wp':
            # 评测改为与训练一致：不显式提供 waypoints，使用 tester 内部固定 RNG=42 生成；线性过渡
            return { 'type': 'random_waypoints','initial_xyz':[0,0,0.8], 'params': {'hold_time':1.2, 'transition':'linear'}}
        if name == 'spiral_in_out':
            return { 'type': 'spiral_in_out','initial_xyz':[0,0,0.7], 'params': {'R_in':0.9,'R_out':0.2,'period':14,'z_wave':0.15}}
        if name == 'stairs':
            # 与训练脚本对齐：起始高度/楼层列表
            return { 'type': 'stairs','initial_xyz':[0,0,0.6], 'params': {'levels':[0.6,0.9,1.2], 'segment_time':3.0}}
        if name == 'coupled_surface':
            return { 'type': 'coupled_surface','initial_xyz':[0,0,0.8], 'params': {'ax':0.9,'ay':0.7,'f1':1.0,'f2':2.0,'phase':1.047,'z_amp':0.25,'surf_amp':0.15}}
        raise ValueError(f"Unknown trajectory: {name}")

    _preset_map = {
        'train_core': ['figure8','helix','circle','square','step_hover','spiral_out'],
        'test_challenge': ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
        'full_eval': ['figure8','helix','circle','square','step_hover','spiral_out',
                      'zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
        # 强化 PI-Light 优势的训练/测试预设（突出非平稳/多相位），两者当前相同但语义区分
        'pi_strong_train': ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
        'pi_strong_test':  ['zigzag3d','lemniscate3d','random_wp','spiral_in_out','stairs','coupled_surface'],
    }

    def _resolve_traj_names(manual_list, preset, fallback_single):
        if manual_list:
            return list(manual_list)
        if preset:
            names = _preset_map.get(preset, [])
            if not names:
                raise ValueError(f"未知轨迹预设: {preset}")
            print(f"[INFO] 使用轨迹预设 {preset}: {names}")
            # 去重同时保持顺序
            return list(dict.fromkeys(names))
        if fallback_single is None:
            return []
        return [fallback_single]

    traj_names = _resolve_traj_names(args.traj_list, getattr(args, 'traj_preset', None), args.trajectory)
    trajectories = [build_trajectory(n) for n in traj_names]
    test_traj_names = _resolve_traj_names(args.test_traj_list, getattr(args, 'test_traj_preset', None), None) if (args.test_traj_list or getattr(args, 'test_traj_preset', None)) else []
    test_aggregate = args.test_aggregate if args.test_aggregate else args.aggregate
    print(f"[INFO] 训练/主评估轨迹(Train): {traj_names} aggregate={args.aggregate}")
    if any(n == 'random_wp' for n in (traj_names + test_traj_names)):
        print(f"[INFO] 随机轨迹生成种子: {traj_seed}")
    if test_traj_names:
        print(f"[INFO] 测试集轨迹(Test): {test_traj_names} test_aggregate={test_aggregate}")
    # ---- 扰动场景：与训练脚本 build_disturbances 对齐 ----
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
    trajectory_test_scenarios = build_disturbances(getattr(args,'disturbance', None))
    # 奖励配置通过 profile 获取（含权重与 k 系数；当前 tester 仅用 weights，k_* 仍在 reward.py 将来可透传）
    reward_weights, reward_ks = get_reward_profile(args.reward_profile)
    print(describe_profile(args.reward_profile))

    best_program_found = []
    if args.mode in ['compare_all','pi_only']:
        # 仅使用固定默认路径
        if os.path.isfile(DEFAULT_PI_PROGRAM):
            try:
                if load_program_json is None:
                    raise RuntimeError('load_program_json 未正确加载')
                best_program_found = load_program_json(DEFAULT_PI_PROGRAM)  # type: ignore
                print(f"[加载] 已载入 Pi-Light 程序(默认路径): {DEFAULT_PI_PROGRAM} (规则数={len(best_program_found)})")
            except Exception as e:
                print(f"[警告] 载入默认 Pi-Light 程序失败: {e}")
        else:
            print(f"[提示] 未找到默认 Pi-Light 程序文件 {DEFAULT_PI_PROGRAM}，将跳过 Pi-Light 评估。")

    summary_records = []

    # 预先初始化测试集分数变量防止静态分析未定义警告
    baseline_test_score = None
    cma_test_score = None
    pi_light_test_score = None
    gsn_test_score = None

    def aggregate_scores(scores):
        if not scores: return None
        if args.aggregate=='mean': return float(sum(scores)/len(scores))
        if args.aggregate=='min': return float(min(scores))
        if args.aggregate=='harmonic':
            import math
            return len(scores)/sum(1/(s+1e-9) for s in scores)
        return float(sum(scores)/len(scores))

    per_traj_results = {}

    def eval_controller(make_controller_fn, label: str, use_test=False):
        scs = []
        indiv = []
        active_trajs = ( [build_trajectory(n) for n in test_traj_names] if use_test else trajectories )
        for t in active_trajs:
            ctrl = make_controller_fn()
            # 各控制器分类存放
            if label == 'baseline_pid':
                out_dir = os.path.join(paths.BASELINE_RESULTS_ROOT,'eval_pid')
            elif label == 'cma_es_pid':
                out_dir = os.path.join(paths.CMAES_RESULTS_ROOT,'eval_pid')
            elif label == 'pi_light_segmented':
                out_dir = os.path.join(paths.PI_FLIGHT_RESULTS_ROOT,'eval_pi_light_segmented')
            elif label == 'gsn':
                out_dir = os.path.join(paths.GSN_RESULTS_ROOT,'eval_gsn')
            elif label == 'attn':
                out_dir = os.path.join(paths.GSN_RESULTS_ROOT,'eval_attn')
            else:
                out_dir = f'results/eval_{label}'  # fallback
            tester = SimulationTester(ctrl, trajectory_test_scenarios, reward_weights,
                                      duration_sec=args.duration_eval, output_folder=out_dir,
                                      gui=False, trajectory=t, log_skip=max(1,int(getattr(args,'log_skip',1))), in_memory=True, quiet=True)
            r = tester.run(); scs.append(r); indiv.append({'traj': t['type'], 'reward': float(r)})
        agg = aggregate_scores(scs)
        if not use_test:
            per_traj_results[label] = indiv
            summary_records.append({'controller': label, 'reward': agg, 'per_traj': indiv})
            tag = 'Train'
        else:
            tag = 'Test'
            # 为测试集在 summary 中补充一行或补充字段
            summary_records.append({'controller': label + '_test', 'reward': agg, 'per_traj': indiv, 'test': True})
        print(f"{label} ({tag}) 聚合得分: {agg:.4f}")
        return agg

    if args.mode in ['compare_all','gsn_only','baseline_only']:
        print("\n[评估] 标准 PID 控制器 (多场景)")
        def make_baseline(): return DSLPIDControl(drone_model=DroneModel("cf2x"))
        baseline_score = eval_controller(make_baseline, 'baseline_pid')
        baseline_test_score = eval_controller(make_baseline, 'baseline_pid', use_test=True) if test_traj_names else None
    else:
        baseline_score = None
        baseline_test_score = None

    if args.mode in ['compare_all']:
        cma_score = None
        best_params = None
        cma_json_path, cma_exists = _pick_cma_json_path()
        if cma_exists:
            try:
                with open(cma_json_path,'r',encoding='utf-8') as f:
                    cma_data = json.load(f)
                # 新格式优先：best_program 风格 -> 从 rules 提取 P/I/D
                if isinstance(cma_data, dict) and 'rules' in cma_data and isinstance(cma_data['rules'], list) and cma_data['rules']:
                    try:
                        rules = cma_data['rules']
                        act = rules[0].get('action', [])
                        p = i = d = 1.0
                        for a in act:
                            if a.get('type') == 'Binary' and a.get('op') == 'set':
                                left = a.get('left', {})
                                right = a.get('right', {})
                                if left.get('type') == 'Terminal' and right.get('type') == 'Terminal':
                                    if left.get('value') == 'P': p = float(right.get('value'))
                                    if left.get('value') == 'I': i = float(right.get('value'))
                                    if left.get('value') == 'D': d = float(right.get('value'))
                        best_params = [p,i,d]
                        print(f"\n[评估] (加载) CMA-ES best_program -> params={best_params} <- {cma_json_path}")
                    except Exception as _bp_e:
                        # 回退到旧格式字段
                        best_params = cma_data.get('best_params',[1.0,1.0,1.0])
                        print(f"\n[评估] (加载) CMA-ES 解析 best_program 失败，回退 best_params，原因: {_bp_e}")
                else:
                    best_params = cma_data.get('best_params',[1.0,1.0,1.0])
                    print(f"\n[评估] (加载) CMA-ES PID 控制器 params={best_params} <- {cma_json_path}")
            except Exception as e:
                print(f"[警告] 读取 CMA-ES 结果失败: {e}")
        elif args.run_cma:
            try:
                import cma
                print(f"[运行] 内联 CMA-ES 搜索 (iters={args.cma_iters}, pop={args.cma_pop}) ...")
                x0=[1.0,1.0,1.0]; sigma0=0.3; bounds=[[0.3,0.0,0.0],[3.0,1.5,2.0]]
                es=cma.CMAEvolutionStrategy(x0, sigma0, {'bounds':bounds,'popsize':args.cma_pop,'maxiter':args.cma_iters})
                gen_hist=[]
                # 说明：内联 CMA-ES 每个候选仅在第一条轨迹上评估以控制速度；如需多场景可改为循环 trajectories
                primary_traj = trajectories[0]
                while not es.stop():
                    sols=es.ask(); losses=[]
                    for sol in sols:
                        ctrl=DSLPIDControl(drone_model=DroneModel("cf2x"))
                        p,i,d=sol
                        ctrl.P_COEFF_TOR[:]*=p; ctrl.I_COEFF_TOR[:]*=i; ctrl.D_COEFF_TOR[:]*=d
                        tester_tmp=SimulationTester(ctrl, trajectory_test_scenarios, reward_weights, duration_sec=args.duration_eval, output_folder=os.path.join('01_pi_flight','results','tmp_cma_inline'), gui=False, trajectory=primary_traj)
                        rew=tester_tmp.run(); gen_hist.append({'params':list(map(float,sol)),'reward':float(rew)}); losses.append(-rew)
                    es.tell(sols,losses)
                best_params=es.result.xbest; best_reward=-es.result.fbest
                os.makedirs(os.path.dirname(cma_json_path), exist_ok=True)
                with open(cma_json_path,'w',encoding='utf-8') as f:
                    json.dump({'best_params':list(map(float,best_params)),'best_reward':float(best_reward),'history':gen_hist}, f, indent=2, ensure_ascii=False)
                print(f"[完成] 内联 CMA-ES best_reward={best_reward:.4f} params={best_params} -> {cma_json_path}")
            except ImportError:
                print('[警告] 未安装 cma 库，跳过内联 CMA-ES。')
            except Exception as e:
                print(f"[警告] 内联 CMA-ES 失败: {e}")
        if best_params is not None:
            def make_cma():
                ctrl = DSLPIDControl(drone_model=DroneModel("cf2x"))
                if best_params is not None:
                    p_gain,i_gain,d_gain = best_params
                    ctrl.P_COEFF_TOR[:] *= p_gain
                    ctrl.I_COEFF_TOR[:] *= i_gain
                    ctrl.D_COEFF_TOR[:] *= d_gain
                return ctrl
            print("\n[评估] CMA-ES PID (多场景)")
            cma_score = eval_controller(make_cma, 'cma_es_pid')
            cma_test_score = eval_controller(make_cma, 'cma_es_pid', use_test=True) if test_traj_names else None
            # 把参数补到 summary 最后一条
            summary_records[-1]['params'] = list(map(float,best_params))
            summary_records[-1]['inline'] = (not cma_exists or args.run_cma)
    else:
        cma_score = None
        cma_test_score = None

    if args.mode in ['compare_all','pi_only'] and best_program_found:
        print("\n[评估] π-Light 分段 PID 控制器 (多场景)")
        captured_rule_hits = {}
        def make_pi():
            if PiLightSegmentedPIDController is None:
                raise RuntimeError('PiLightSegmentedPIDController 未加载')
            ctrl = PiLightSegmentedPIDController(
                drone_model=DroneModel("cf2x"), program=best_program_found,
                compose_by_gain=bool(getattr(args, 'compose_by_gain', False)),
                clip_P=getattr(args,'clip_P',None), clip_I=getattr(args,'clip_I',None), clip_D=getattr(args,'clip_D',None),
                semantics=getattr(args,'semantics', None), require_k=int(getattr(args,'require_k',0) or 0), blend_topk_k=int(getattr(args,'blend_topk_k',2) or 2),
                gain_slew_limit=getattr(args,'gain_slew_limit', None), min_hold_steps=int(getattr(args,'min_hold_steps',0) or 0)
            )  # type: ignore
            return ctrl
        pi_light_score = eval_controller(make_pi, 'pi_light_segmented')
        pi_light_test_score = eval_controller(make_pi, 'pi_light_segmented', use_test=True) if test_traj_names else None
        # eval_controller doesn't expose controller; for aggregated stats we'd need integration inside loop. Placeholder: None.
        summary_records[-1]['rules'] = len(best_program_found)
    else:
        pi_light_score = None
        pi_light_test_score = None

    if args.mode in ['gsn_only','compare_all']:
        # Directly use numbered folder; ensure it's on sys.path
        numbered_dir = os.path.join(os.path.dirname(__file__), '04_nn_baselines')
        if os.path.isdir(numbered_dir) and numbered_dir not in sys.path:
            sys.path.insert(0, numbered_dir)
        try:
            from gsn_controller import GSNController  # type: ignore
        except Exception as _e:
            raise ImportError("无法导入 GSNController (04_nn_baselines/gsn_controller.py)") from _e
        import torch
        print("\n[评估] GSN 增益调度控制器 (多场景)")
        state_dim = 20
        legacy_dataset = os.path.join('results','gsn_dataset.npz')
        if os.path.exists(legacy_dataset):
            try:
                d = np.load(legacy_dataset, allow_pickle=True)
                meta = eval(d['meta'][0])
                state_dim = meta.get('state_dim', state_dim)
            except Exception:
                pass
        def make_gsn():
            gsn_ctrl = GSNController(drone_model=DroneModel("cf2x"), state_dim=state_dim)
            if os.path.exists(args.gsn_ckpt):
                ckpt = torch.load(args.gsn_ckpt, map_location='cpu')
                gsn_ctrl.model.load_state_dict(ckpt['model'])
            return gsn_ctrl
        gsn_score = eval_controller(make_gsn, 'gsn')
        gsn_test_score = eval_controller(make_gsn, 'gsn', use_test=True) if test_traj_names else None
        summary_records[-1]['state_dim']=state_dim
        summary_records[-1]['ckpt']=args.gsn_ckpt
        if not os.path.exists(args.gsn_ckpt):
            print(f"[警告] 未找到 GSN 权重 {args.gsn_ckpt}，使用随机初始化模型。")
    else:
        gsn_score = None
        gsn_test_score = None

    # Attention baseline
    if args.mode in ['attn_only','compare_all']:
        numbered_dir = os.path.join(os.path.dirname(__file__), '04_nn_baselines')
        if os.path.isdir(numbered_dir) and numbered_dir not in sys.path:
            sys.path.insert(0, numbered_dir)
        try:
            from attn_controller import AttnController  # type: ignore
        except Exception as _e:
            raise ImportError("无法导入 AttnController (04_nn_baselines/attn_controller.py)") from _e
        import torch  # noqa: F401
        print("\n[评估] Attention 增益调度控制器 (多场景)")
        state_dim = 20
        def make_attn():
            ctrl = AttnController(drone_model=DroneModel("cf2x"), state_dim=state_dim)
            if os.path.exists(args.attn_ckpt):
                ckpt = torch.load(args.attn_ckpt, map_location='cpu')
                ctrl.model.load_state_dict(ckpt['model'])
            return ctrl
        attn_score = eval_controller(make_attn, 'attn')
        attn_test_score = eval_controller(make_attn, 'attn', use_test=True) if test_traj_names else None
        summary_records[-1]['state_dim'] = state_dim
        summary_records[-1]['ckpt'] = args.attn_ckpt
        if not os.path.exists(args.attn_ckpt):
            print(f"[警告] 未找到 Attention 权重 {args.attn_ckpt}，使用随机初始化模型。")
    else:
        attn_score = None
        attn_test_score = None

    print("\n================= 汇总结果 (无 GUI 多场景) =================")
    def fmt_pair(name, train, test):
        if test is None:
            return f"{name:<11}: {train}"
        return f"{name:<11}: {train:.4f}  | Test: {test:.4f}"
    if baseline_score is not None: print(fmt_pair('Baseline PID', baseline_score, baseline_test_score))
    if cma_score is not None: print(fmt_pair('CMA-ES PID', cma_score, cma_test_score))
    if pi_light_score is not None: print(fmt_pair('Pi-Light', pi_light_score, pi_light_test_score))
    if gsn_score is not None: print(fmt_pair('GSN', gsn_score, gsn_test_score))
    if attn_score is not None: print(fmt_pair('ATTN', attn_score, attn_test_score))
    print("训练轨迹集合:", traj_names)
    if test_traj_names:
        print("测试轨迹集合:", test_traj_names)
    print("==================================================")

    if summary_records and args.save_summary:
        try:
            os.makedirs(os.path.dirname(args.save_summary), exist_ok=True)
            payload = {
                'summary':summary_records,
                'train_trajectories':traj_names,
                'test_trajectories':test_traj_names,
                'aggregate':args.aggregate,
                'test_aggregate': test_aggregate,
                'disturbance': getattr(args,'disturbance', None),
                'traj_seed': traj_seed,
                'duration_eval':args.duration_eval,
                'headless': True,
                'argv':' '.join(sys.argv)
            }
            with open(args.save_summary,'w',encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            ts_path = os.path.join(paths.SUMMARY_DIR, f"summary_{paths.timestamp()}.json")
            with open(ts_path,'w',encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"[保存] 汇总结果 -> {args.save_summary} 以及 {ts_path}")
        except Exception as e:
            print(f"[警告] 保存 summary 失败: {e}")







