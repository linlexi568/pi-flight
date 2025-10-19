import time, os, sys, numpy as np, argparse, json, importlib.util
from typing import Any, cast
from utilities.reward_profiles import get_reward_profile, describe_profile
from utilities import paths

# ---- 动态加载 PI-Flight (优先 01_pi_flight 或包 pi_flight；兼容旧包 pi_light) ----
PiLightSegmentedPIDController = None  # type: ignore
load_program_json = None  # type: ignore
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_PF_DIR = os.path.join(_ROOT_DIR, '01_pi_flight')
def _load_pilight():
    global PiLightSegmentedPIDController, load_program_json
    # 先尝试新包名
    try:
        from pi_flight import PiLightSegmentedPIDController as _P, serialization as _S  # type: ignore
        PiLightSegmentedPIDController = _P
        load_program_json = getattr(_S, 'load_program_json', None)
        return
    except Exception:
        pass
    # 回退尝试旧包名
    try:
        from pi_light import PiLightSegmentedPIDController as _P, serialization as _S  # type: ignore
        PiLightSegmentedPIDController = _P
        load_program_json = getattr(_S, 'load_program_json', None)
        return
    except Exception:
        pass
    # 最后尝试编号目录 01_pi_flight
    init_file = os.path.join(_PF_DIR, '__init__.py')
    ser_file = os.path.join(_PF_DIR, 'serialization.py')
    if os.path.isfile(init_file):
        spec = importlib.util.spec_from_file_location('piflight_dynamic', init_file, submodule_search_locations=[_PF_DIR])
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules['piflight_dynamic'] = module
            try:
                spec.loader.exec_module(module)  # type: ignore
                PiLightSegmentedPIDController = getattr(module, 'PiLightSegmentedPIDController', None)
            except Exception as e:
                print(f"[Warn] 载入 01_pi_flight/__init__.py 失败: {e}")
    if load_program_json is None and os.path.isfile(ser_file):
        spec2 = importlib.util.spec_from_file_location('piflight_dynamic.serialization', ser_file)
        if spec2 and spec2.loader:
            mod2 = importlib.util.module_from_spec(spec2)
            sys.modules['piflight_dynamic.serialization'] = mod2
            try:
                spec2.loader.exec_module(mod2)  # type: ignore
                load_program_json = getattr(mod2, 'load_program_json', None)
            except Exception as e:
                print(f"[Warn] 载入 01_pi_flight/serialization 失败: {e}")
    if PiLightSegmentedPIDController is None or load_program_json is None:
        raise ImportError("无法加载 PI-Flight 组件 (尝试 01_pi_flight 或包 pi_flight/pi_light)。")
_load_pilight()
from test import SimulationTester
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
assert PiLightSegmentedPIDController is not None and load_program_json is not None
# Local aliases with permissive typing to satisfy static analysis
_PLC = cast(Any, PiLightSegmentedPIDController)
_LPJ = cast(Any, load_program_json)

def create_evaluation_function(test_scenarios: list, weights: dict, trajectory: dict, duration_sec: int):
    """创建一个函数，该函数接收一个程序，返回其奖励分数。"""
    def evaluate(program: list):
        # print(f"\n--- [评估开始] ---")
        # start_time = time.time()
        
        # 使用新的分段控制器
        controller = _PLC(
            drone_model=DroneModel("cf2x"),
            program=program
        )
        
        tester = SimulationTester(
            controller=controller,
            test_scenarios=test_scenarios,
            output_folder=os.path.join('01_pi_flight','results','mcts_eval'),
            gui=False, # MCTS 搜索期间禁用GUI以提高速度
            weights=weights,
            trajectory=trajectory,
            duration_sec=duration_sec
        )
        
        reward = tester.run()
        # end_time = time.time()
        # print(f"--- [评估结束] 程序获得奖励: {reward:.4f}, 耗时: {end_time - start_time:.2f}s ---")
        return reward
    return evaluate

# 在文件顶部添加参数解析（保持原有逻辑可作为默认）

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, default='compare_all', choices=['compare_all','gsn_only','baseline_only','pi_only'], help='运行模式')
    ap.add_argument('--gsn_ckpt', type=str, default=os.path.join('04_nn_baselines','results','checkpoints','gsn_best.pt'), help='GSN 模型权重路径 (legacy: nn_baselines/results/...)')
    ap.add_argument('--pi_program', type=str, default=os.path.join('01_pi_flight','results','best_program.json'), help='外部训练好的 PI-Flight 程序 JSON 路径 (legacy: pi_light/results/...)')
    ap.add_argument('--duration_eval', type=int, default=20, help='评估阶段仿真时长(s)')
    ap.add_argument('--trajectory', type=str, default='figure8', choices=['figure8','helix'])
    ap.add_argument('--no_gui', action='store_true')
    ap.add_argument('--cma_json', type=str, default=os.path.join('CMA-ES','results','cma_es_pid_result.json'), help='CMA-ES 最优结果 JSON 文件路径 (若存在直接加载)')
    ap.add_argument('--run_cma', action='store_true', help='若指定且 cma_json 不存在，则在线运行一次简化 CMA-ES 搜索')
    ap.add_argument('--cma_iters', type=int, default=15, help='内联 CMA-ES 最大迭代数 (近似 generation)')
    ap.add_argument('--cma_pop', type=int, default=8, help='内联 CMA-ES 种群大小')
    ap.add_argument('--save_summary', type=str, default=os.path.join(paths.SUMMARY_DIR,'summary_latest.json'), help='保存汇总 JSON 路径')
    ap.add_argument('--reward_profile', type=str, default='pilight_boost', choices=['default','pilight_boost','pilight_freq_boost'], help='奖励权重/系数配置档 (default|pilight_boost|pilight_freq_boost)')
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Legacy -> numbered folder rewrites
    def _rewrite(pth: str, old_prefix: str, new_prefix: str, flag: str):
        if pth.startswith(old_prefix):
            tail = pth[len(old_prefix):].lstrip('/\\')
            new_path = os.path.join(new_prefix, tail)
            print(f"[LegacyPathWarning] {flag}: '{pth}' -> '{new_path}'")
            return new_path
        return pth
    # results/pi_light -> 01_pi_flight/results
    args.pi_program = _rewrite(args.pi_program, os.path.join('results','pi_light'), os.path.join('01_pi_flight','results'), '--pi_program')
    # pi_light/results -> 01_pi_flight/results
    args.pi_program = _rewrite(args.pi_program, os.path.join('pi_light','results'), os.path.join('01_pi_flight','results'), '--pi_program')
    # nn_baselines -> 04_nn_baselines
    args.gsn_ckpt = _rewrite(args.gsn_ckpt, 'nn_baselines', '04_nn_baselines', '--gsn_ckpt')
    # CMA-ES results path -> 03_CMA-ES/results
    args.cma_json = _rewrite(args.cma_json, os.path.join('CMA-ES','results'), os.path.join('03_CMA-ES','results'), '--cma_json')
    # 定义 MCTS 搜索空间（确保在任何分支下可用）
    DSL_VARIABLES = [
        'err_p_roll', 'err_p_pitch', 'err_d_roll', 'err_d_pitch',
        'ang_vel_x', 'ang_vel_y', 'err_i_roll', 'err_i_pitch',
        'pos_err_x', 'pos_err_y', 'pos_err_z',
        'err_i_x', 'err_i_y', 'err_i_z'
    ]
    DSL_CONSTANTS = [0.1, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 5.0, 10.0]
    DSL_OPERATORS = ['+', '-', '*', 'abs', '>', '<', 'max', 'min', 'sin', 'cos']
    # 选择轨迹
    trajectory_figure_8 = { 'type': 'figure_8','initial_xyz': [0, 0, 1.0], 'params': {'A': 0.8,'B': 0.5,'period': 12}}
    trajectory_helix = { 'type': 'helix','initial_xyz': [0, 0, 0.5], 'params': {'R': 0.7,'period': 10,'v_z': 0.15}}
    active_trajectory = trajectory_figure_8 if args.trajectory=='figure8' else trajectory_helix
    # 奖励权重保持不变 (后续可引入 args)
    trajectory_test_scenarios = []
    # 使用 reward_profile 获取权重
    reward_weights, reward_ks = get_reward_profile(args.reward_profile)
    print(describe_profile(args.reward_profile))

    # 改为外部加载 program，不再内部搜索
    best_program_found = []
    if args.mode in ['compare_all','pi_only']:
        if args.pi_program and os.path.isfile(args.pi_program):
            try:
                best_program_found = _LPJ(args.pi_program)
                print(f"[加载] 已载入 Pi-Light 程序: {args.pi_program} (规则数={len(best_program_found)})")
            except Exception as e:
                print(f"[警告] 载入 Pi-Light 程序失败: {e}")
        else:
            print(f"[提示] 未找到 Pi-Light 程序文件 {args.pi_program}，将跳过 Pi-Light 评估。")

    summary_records = []

    # 基线 PID
    if args.mode in ['compare_all','gsn_only','baseline_only']:
        print("\n[评估] 标准 PID 控制器")
        baseline_pid_controller = DSLPIDControl(drone_model=DroneModel("cf2x"))
        baseline_out = os.path.join(paths.BASELINE_RESULTS_ROOT,'eval_pid')
        baseline_tester = SimulationTester(baseline_pid_controller, trajectory_test_scenarios, reward_weights, duration_sec=args.duration_eval, output_folder=baseline_out, gui=not args.no_gui, trajectory=active_trajectory)
        baseline_score = baseline_tester.run()
        print(f"标准 PID 得分: {baseline_score:.4f}")
        summary_records.append({'controller':'baseline_pid','reward':float(baseline_score)})
    else:
        baseline_score = None

    # CMA-ES PID (载入最优缩放参数并复现)
    if args.mode in ['compare_all']:
        cma_score = None
        best_params = None
        if os.path.isfile(args.cma_json):
            try:
                with open(args.cma_json,'r',encoding='utf-8') as f:
                    cma_data = json.load(f)
                best_params = cma_data.get('best_params',[1.0,1.0,1.0])
                print(f"\n[评估] (加载) CMA-ES PID 控制器 params={best_params}")
            except Exception as e:
                print(f"[警告] 读取 CMA-ES 结果失败: {e}")
        elif args.run_cma:
            # 轻量内联 CMA-ES 搜索 (不依赖外部脚本)；若希望更强可运行独立脚本
            try:
                import cma
                print(f"[运行] 内联 CMA-ES 搜索 (iters={args.cma_iters}, pop={args.cma_pop}) ...")
                x0 = [1.0,1.0,1.0]; sigma0 = 0.3
                bounds = [[0.3,0.0,0.0],[3.0,1.5,2.0]]
                es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds':bounds,'popsize':args.cma_pop,'maxiter':args.cma_iters})
                traj = active_trajectory
                gen_hist = []
                while not es.stop():
                    solutions = es.ask()
                    losses = []
                    for sol in solutions:
                        ctrl = DSLPIDControl(drone_model=DroneModel("cf2x"))
                        p,i,d = sol
                        ctrl.P_COEFF_TOR[:] *= p; ctrl.I_COEFF_TOR[:] *= i; ctrl.D_COEFF_TOR[:] *= d
                        tester_tmp = SimulationTester(ctrl, trajectory_test_scenarios, reward_weights, duration_sec=args.duration_eval, output_folder=os.path.join('01_pi_flight','results','tmp_cma_inline'), gui=not args.no_gui, trajectory=traj)
                        rew = tester_tmp.run()
                        gen_hist.append({'params':list(map(float,sol)),'reward':float(rew)})
                        losses.append(-rew)
                    es.tell(solutions, losses)
                best_params = es.result.xbest
                best_reward = -es.result.fbest
                # 保存结果
                os.makedirs(os.path.dirname(args.cma_json), exist_ok=True)
                with open(args.cma_json,'w',encoding='utf-8') as f:
                    json.dump({'best_params':list(map(float,best_params)),'best_reward':float(best_reward),'history':gen_hist}, f, indent=2, ensure_ascii=False)
                print(f"[完成] 内联 CMA-ES best_reward={best_reward:.4f} params={best_params} -> {args.cma_json}")
            except ImportError:
                print('[警告] 未安装 cma 库，跳过内联 CMA-ES；可 pip install cma 后重试。')
            except Exception as e:
                print(f"[警告] 内联 CMA-ES 失败: {e}")
        if best_params is not None:
            cma_pid_controller = DSLPIDControl(drone_model=DroneModel("cf2x"))
            p_gain,i_gain,d_gain = best_params
            cma_pid_controller.P_COEFF_TOR[:] *= p_gain
            cma_pid_controller.I_COEFF_TOR[:] *= i_gain
            cma_pid_controller.D_COEFF_TOR[:] *= d_gain
            cma_out = os.path.join(paths.CMAES_RESULTS_ROOT,'eval_pid')
            cma_tester = SimulationTester(cma_pid_controller, trajectory_test_scenarios, reward_weights, duration_sec=args.duration_eval, output_folder=cma_out, gui=not args.no_gui, trajectory=active_trajectory)
            cma_score = cma_tester.run()
            print(f"CMA-ES PID 得分: {cma_score:.4f}")
            summary_records.append({'controller':'cma_es_pid','reward':float(cma_score),'params':list(map(float,best_params)),'inline': (not os.path.isfile(args.cma_json) or args.run_cma)})
    else:
        cma_score = None

    # π-Light 分段 PID
    if args.mode in ['compare_all','pi_only'] and best_program_found:
        print("\n[评估] PI-Flight 分段 PID 控制器")
        pi_light_controller = _PLC(drone_model=DroneModel("cf2x"), program=best_program_found)
        pi_light_out = os.path.join(paths.PI_FLIGHT_RESULTS_ROOT,'eval_pi_light_segmented')
        pi_light_tester = SimulationTester(pi_light_controller, trajectory_test_scenarios, reward_weights, duration_sec=args.duration_eval, output_folder=pi_light_out, gui=not args.no_gui, trajectory=active_trajectory)
        pi_light_score = pi_light_tester.run()
        rule_hits = {}
        try:
            rule_hits = pi_light_controller.dump_rule_stats()
        except Exception:
            pass
        print(f"PI-Flight 得分: {pi_light_score:.4f} 规则触发统计: {rule_hits}")
        summary_records.append({'controller':'pi_light_segmented','reward':float(pi_light_score),'rules':len(best_program_found),'rule_hits':rule_hits})
    else:
        pi_light_score = None

    # GSN 基线
    if args.mode in ['gsn_only','compare_all']:
        # 动态加载 04_nn_baselines/gsn_controller.py
        gsn_dir = os.path.join(_ROOT_DIR, '04_nn_baselines')
        gsn_file = os.path.join(gsn_dir, 'gsn_controller.py')
        if gsn_dir not in sys.path:
            sys.path.insert(0, gsn_dir)
        GSNController = None  # type: ignore
        try:
            from  gsn_controller import GSNController as _GC  # type: ignore
            GSNController = _GC
        except Exception:
            if os.path.isfile(gsn_file):
                spec_g = importlib.util.spec_from_file_location('gsn_controller_dynamic', gsn_file)
                if spec_g and spec_g.loader:
                    mod_g = importlib.util.module_from_spec(spec_g)
                    sys.modules['gsn_controller_dynamic'] = mod_g
                    try:
                        spec_g.loader.exec_module(mod_g)  # type: ignore
                        GSNController = getattr(mod_g, 'GSNController', None)
                    except Exception as e:
                        print(f"[Warn] 动态加载 gsn_controller 失败: {e}")
        if GSNController is None:
            raise ImportError('无法导入 GSNController (尝试包 nn_baselines 与目录 04_nn_baselines)')
        import torch
        print("\n[评估] GSN 增益调度控制器")
        state_dim = 20
        legacy_dataset = os.path.join('results','gsn_dataset.npz')
        if os.path.exists(legacy_dataset):
            try:
                d = np.load(legacy_dataset, allow_pickle=True)
                meta = eval(d['meta'][0])
                state_dim = meta.get('state_dim', state_dim)
            except Exception:
                pass
        gsn_controller = GSNController(drone_model=DroneModel("cf2x"), state_dim=state_dim)
        if os.path.exists(args.gsn_ckpt):
            ckpt = torch.load(args.gsn_ckpt, map_location='cpu')
            gsn_controller.model.load_state_dict(ckpt['model'])
            print(f"[加载] 已载入 GSN 权重: {args.gsn_ckpt}")
        else:
            print(f"[警告] 未找到 GSN 权重 {args.gsn_ckpt}，使用随机初始化模型。")
        gsn_out = os.path.join(paths.GSN_RESULTS_ROOT,'eval_gsn')
        gsn_tester = SimulationTester(gsn_controller, trajectory_test_scenarios, reward_weights, duration_sec=args.duration_eval, output_folder=gsn_out, gui=not args.no_gui, trajectory=active_trajectory)
        gsn_score = gsn_tester.run()
        print(f"GSN 得分: {gsn_score:.4f}")
        summary_records.append({'controller':'gsn','reward':float(gsn_score),'state_dim':state_dim,'ckpt':args.gsn_ckpt})
    else:
        gsn_score = None

    # 汇总对比
    print("\n================= 汇总结果 =================")
    if baseline_score is not None: print(f"Baseline PID: {baseline_score}")
    if cma_score is not None: print(f"CMA-ES PID : {cma_score}")
    if pi_light_score is not None: print(f"Pi-Light    : {pi_light_score}")
    if gsn_score is not None: print(f"GSN         : {gsn_score}")
    print("==========================================")

    # 保存 summary JSON
    if summary_records and args.save_summary:
        try:
            os.makedirs(os.path.dirname(args.save_summary), exist_ok=True)
            payload = {'summary':summary_records,'trajectory':args.trajectory,'duration_eval':args.duration_eval,'argv': ' '.join(sys.argv)}
            with open(args.save_summary,'w',encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            # 追加时间戳副本
            ts_path = os.path.join(paths.SUMMARY_DIR, f"summary_{paths.timestamp()}.json")
            with open(ts_path,'w',encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"[保存] 汇总结果 -> {args.save_summary} 以及 {ts_path}")
        except Exception as e:
            print(f"[警告] 保存 summary 失败: {e}")






