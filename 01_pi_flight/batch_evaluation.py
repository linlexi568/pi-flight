"""批量程序评估模块 - Isaac Gym GPU并行加速

仅支持Isaac Gym批量并行仿真（512+ 环境）
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import time

# Isaac Gym检测（尝试从本仓库的 vendor 目录加载）
import sys, pathlib, os
ISAAC_GYM_AVAILABLE = False
try:
    # 优先直接导入
    from isaacgym import gymapi  # type: ignore
    ISAAC_GYM_AVAILABLE = True
except Exception:
    # 尝试将 repo 内置路径加入 sys.path
    try:
        _HERE = pathlib.Path(__file__).resolve()
        _PKG_ROOT = _HERE.parent  # 01_pi_flight
        _REPO_ROOT = _PKG_ROOT.parent  # repo root
        _GYM_PY = _REPO_ROOT / 'isaacgym' / 'python'
        if _GYM_PY.exists() and str(_GYM_PY) not in sys.path:
            sys.path.insert(0, str(_GYM_PY))
        from isaacgym import gymapi  # type: ignore
        ISAAC_GYM_AVAILABLE = True
        # 配置必要的环境变量以定位插件信息
        try:
            os.environ.setdefault('GYM_USD_PLUG_INFO_PATH', str(_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64' / 'usd' / 'plugInfo.json'))
        except Exception:
            pass
    except Exception:
        ISAAC_GYM_AVAILABLE = False


# Stepwise 奖励计算器与权重
try:
    from .reward_stepwise import StepwiseRewardCalculator  # type: ignore
except Exception:
    try:
        from reward_stepwise import StepwiseRewardCalculator  # type: ignore
    except Exception:
        StepwiseRewardCalculator = None  # type: ignore
try:
    from utilities.reward_profiles import get_reward_profile  # type: ignore
except Exception:
    get_reward_profile = None  # type: ignore


class BatchEvaluator:
    """批量程序评估器（仅支持Isaac Gym）"""
    
    def __init__(self, 
                 trajectory_config: Dict[str, Any],
                 duration: int = 20,
                 isaac_num_envs: int = 512,
                 device: str = 'cuda:0',
                 replicas_per_program: int = 1,
                 min_steps_frac: float = 0.0,
                 reward_reduction: str = 'sum',
                 reward_profile: str = 'control_law_discovery',
                 strict_no_prior: bool = True,
                 zero_action_penalty: float = 1.5,
                 use_fast_path: bool = True):
        """
        Args:
            trajectory_config: 轨迹配置 {'type': 'figure8', 'params': {...}}
            duration: 仿真时长（秒）
            isaac_num_envs: Isaac Gym并行环境数
            device: GPU设备
            replicas_per_program: evaluate_single 时为同一程序生成多少副本并行评估，取平均
            min_steps_frac: 每次评估至少执行的步数比例（0-1），避免过早 done 提前退出
            reward_reduction: 奖励归约方式：'sum'（步次求和）或 'mean'（步次平均，抵消存活时长偏差）
            reward_profile: 奖励配置文件名称
        """
        # 保险起见：运行期再尝试一次导入
        global ISAAC_GYM_AVAILABLE
        if not ISAAC_GYM_AVAILABLE:
            try:
                from isaacgym import gymapi  # type: ignore
                ISAAC_GYM_AVAILABLE = True
            except Exception:
                # 再尝试 vendor 路径
                try:
                    _HERE = pathlib.Path(__file__).resolve()
                    _PKG_ROOT = _HERE.parent
                    _REPO_ROOT = _PKG_ROOT.parent
                    _GYM_PY = _REPO_ROOT / 'isaacgym' / 'python'
                    if _GYM_PY.exists() and str(_GYM_PY) not in sys.path:
                        sys.path.insert(0, str(_GYM_PY))
                    from isaacgym import gymapi  # type: ignore
                    os.environ.setdefault('GYM_USD_PLUG_INFO_PATH', str(_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64' / 'usd' / 'plugInfo.json'))
                    ISAAC_GYM_AVAILABLE = True
                except Exception:
                    ISAAC_GYM_AVAILABLE = False
        # 不在此处硬性失败；在真正创建环境时再进行检测并报错
        
        self.trajectory_config = trajectory_config
        self.duration = duration
        self.isaac_num_envs = isaac_num_envs
        self.device = device
        self.replicas_per_program = max(1, int(replicas_per_program))
        self.min_steps_frac = float(min_steps_frac) if 0.0 <= float(min_steps_frac) <= 1.0 else 0.0
        self.reward_reduction = reward_reduction if reward_reduction in ('sum', 'mean') else 'sum'
        self.reward_profile = reward_profile
        # 严格无先验（默认开启）：强制使用直接 u_* 动作路径，完全不依赖内置 PID 框架
        self.strict_no_prior = bool(strict_no_prior)
        # 对整集始终为“零动作”的程序加罚，避免搜索停留在空程序
        try:
            self.zero_action_penalty = float(zero_action_penalty)
        except Exception:
            self.zero_action_penalty = 1.5
        
        # 初始化 Stepwise 奖励计算器（使用 control_law_discovery 权重）
        try:
            weights, ks = get_reward_profile(self.reward_profile)
            # 估计 dt: Isaac 默认物理频率 240 Hz，控制频率 48 Hz -> dt ≈ 1/48
            self._step_dt = 1.0 / 48.0
            self._step_reward_calc = StepwiseRewardCalculator(weights, ks, dt=self._step_dt, num_envs=self.isaac_num_envs, device=self.device)
        except Exception:
            self._step_reward_calc = None

        # Isaac Gym环境池（延迟初始化）
        self._isaac_env_pool = None
        self._envs_ready = False  # 环境池持久化标记
        self._last_reset_size = 0  # 上次reset的环境数
        
        # 🚀 快速路径优化
        self.use_fast_path = use_fast_path
        self._program_cache = {}  # 预编译缓存: {prog_hash: (fz,tx,ty,tz)}
        
        # 🚀🚀 超高性能执行器 (完全向量化 + JIT)
        if use_fast_path:
            try:
                from .ultra_fast_executor import UltraFastExecutor
                self._ultra_executor = UltraFastExecutor()
            except Exception as e:
                try:
                    from ultra_fast_executor import UltraFastExecutor
                    self._ultra_executor = UltraFastExecutor()
                except Exception:
                    print(f"[BatchEvaluator] ⚠️ 超高性能执行器加载失败: {e}")
                    self._ultra_executor = None
        else:
            self._ultra_executor = None
        
        print(f"[BatchEvaluator] 初始化完成")
        print(f"  - Isaac Gym: {'✅ 启用' if ISAAC_GYM_AVAILABLE else '❌ 未启用'}")
        print(f"  - 并行环境数: {self.isaac_num_envs}")
        print(f"  - GPU设备: {self.device}")
        print(f"  - 单程序副本数: {self.replicas_per_program}")
        print(f"  - 最小步数比例: {self.min_steps_frac}")
        print(f"  - 奖励归约: {self.reward_reduction}")
        print(f"  - 严格无先验(u_*直接控制): {'✅ 是' if self.strict_no_prior else '❌ 否'}")
        if self.strict_no_prior:
            print(f"  - 零动作惩罚: {self.zero_action_penalty}")
    
    def _init_isaac_gym_pool(self):
        """延迟初始化Isaac Gym环境池"""
        if self._isaac_env_pool is not None:
            return
        
        print(f"[BatchEvaluator] 初始化Isaac Gym环境池...")
        
        # 导入Isaac Gym环境
        try:
            from .envs.isaac_gym_drone_env import IsaacGymDroneEnv
        except ImportError:
            try:
                from envs.isaac_gym_drone_env import IsaacGymDroneEnv
            except ImportError:
                raise ImportError("无法导入IsaacGymDroneEnv，请检查envs目录")
        # 控制器
        try:
            from .segmented_controller import PiLightSegmentedPIDController
        except ImportError:
            try:
                from segmented_controller import PiLightSegmentedPIDController
            except ImportError:
                PiLightSegmentedPIDController = None  # type: ignore
        
        # 创建环境池
        self._isaac_env_pool = IsaacGymDroneEnv(
            num_envs=self.isaac_num_envs,
            device=self.device,
            headless=True,
            duration_sec=self.duration
        )
        # 保存控制周期
        try:
            self._control_freq = int(self._isaac_env_pool.control_freq)
        except Exception:
            self._control_freq = 48
        self._control_dt = 1.0 / float(self._control_freq)
        
        print(f"[BatchEvaluator] ✅ Isaac Gym环境池就绪（{self.isaac_num_envs} 环境）")

    # ---------------------- DSL 辅助：AST 求值与动作解析 ----------------------
    def _ast_eval(self, node, state: Dict[str, float]) -> float:
        """最小求值器：支持 MCTS 生成的算子集（数值表达式）。"""
        try:
            # 延迟导入 DSL 结点类型
            try:
                from .dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore
            except Exception:
                from dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore

            # 递归求值
            if isinstance(node, (int, float)):
                return float(node)
            # 终端：变量名或常数
            if hasattr(node, 'value') and not hasattr(node, 'op'):
                v = getattr(node, 'value', 0.0)
                if isinstance(v, str):
                    return float(state.get(v, 0.0))
                return float(v)
            # 一元
            if hasattr(node, 'op') and hasattr(node, 'child'):
                x = float(self._ast_eval(node.child, state))
                op = str(getattr(node, 'op', ''))
                if op == 'abs':
                    return abs(x)
                if op == 'sin':
                    import math
                    return float(math.sin(x))
                if op == 'cos':
                    import math
                    return float(math.cos(x))
                if op == 'tan':
                    import math
                    return float(max(-10.0, min(10.0, math.tan(x))))
                if op == 'log1p':
                    import math
                    return float(math.log1p(abs(x)))
                if op == 'sqrt':
                    import math
                    return float(math.sqrt(abs(x)))
                if op == 'sign':
                    return float(1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
                return float(x)
            # 二元
            if hasattr(node, 'op') and hasattr(node, 'left') and hasattr(node, 'right'):
                op = str(getattr(node, 'op', ''))
                if op in ('+', '-', '*', '/', 'max', 'min'):
                    a = float(self._ast_eval(node.left, state))
                    b = float(self._ast_eval(node.right, state))
                    if op == '+':
                        return a + b
                    if op == '-':
                        return a - b
                    if op == '*':
                        return a * b
                    if op == '/':
                        return a / b if abs(b) > 1e-9 else (a * 1.0)
                    if op == 'max':
                        return a if a >= b else b
                    if op == 'min':
                        return a if a <= b else b
                elif op in ('<', '>', '==', '!='):
                    a = float(self._ast_eval(node.left, state))
                    b = float(self._ast_eval(node.right, state))
                    if op == '<':
                        return 1.0 if a < b else 0.0
                    if op == '>':
                        return 1.0 if a > b else 0.0
                    if op == '==':
                        return 1.0 if abs(a - b) < 1e-9 else 0.0
                    if op == '!=':
                        return 1.0 if abs(a - b) >= 1e-9 else 0.0
            # IfNode
            if hasattr(node, 'condition') and hasattr(node, 'then_branch') and hasattr(node, 'else_branch'):
                c = float(self._ast_eval(node.condition, state))
                return float(self._ast_eval(node.then_branch if c > 0 else node.else_branch, state))
        except Exception:
            pass
        return 0.0

    def _program_uses_u(self, program: List[Dict[str, Any]]) -> bool:
        """检测动作是否使用了 u_fz/u_tx/u_ty/u_tz 键。"""
        try:
            for rule in program or []:
                acts = rule.get('action', []) or []
                for a in acts:
                    try:
                        # a 为 BinaryOpNode('set', TerminalNode(key), expr)
                        if hasattr(a, 'op') and a.op == 'set' and hasattr(a, 'left') and hasattr(a.left, 'value'):
                            key = str(getattr(a.left, 'value', ''))
                            if key in ('u_fz', 'u_tx', 'u_ty', 'u_tz'):
                                return True
                    except Exception:
                        continue
        except Exception:
            return False
        return False

    def _compile_program_fast(self, program: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        """
        🚀 快速路径: 预编译常量程序 (u_fz/u_tx/u_ty/u_tz = const)
        
        对于简单的常量控制程序,直接提取常量值,避免重复AST求值
        """
        fz = tx = ty = tz = 0.0
        for rule in program or []:
            if rule.get('op') == 'set':
                var = rule.get('var', '')
                expr = rule.get('expr', {})
                if expr.get('type') == 'const':
                    val = float(expr.get('value', 0.0))
                    if var == 'u_fz':
                        fz = val
                    elif var == 'u_tx':
                        tx = val
                    elif var == 'u_ty':
                        ty = val
                    elif var == 'u_tz':
                        tz = val
        # 裁剪
        fz = float(max(-5.0, min(5.0, fz)))
        tx = float(max(-0.02, min(0.02, tx)))
        ty = float(max(-0.02, min(0.02, ty)))
        tz = float(max(-0.01, min(0.01, tz)))
        return fz, tx, ty, tz
    
    def _eval_program_forces(self, program: List[Dict[str, Any]], state: Dict[str, float]) -> Tuple[float, float, float, float]:
        """在给定数值 state 下，求解程序产生的 (fz, tx, ty, tz)。
        策略：聚合所有满足条件的规则，将 set 的值累加（可适度裁剪）。
        """
        # 🚀 快速路径: 如果启用且程序在缓存中
        if self.use_fast_path:
            try:
                # 程序哈希 (简化: 用str表示)
                prog_str = str([(r.get('op'), r.get('var'), r.get('expr')) for r in program])
                if prog_str in self._program_cache:
                    return self._program_cache[prog_str]
                
                # 尝试快速编译
                result = self._compile_program_fast(program)
                self._program_cache[prog_str] = result
                
                # 调试: 首次缓存 (减少日志)
                # if len(self._program_cache) <= 5:
                #     print(f"[FastPath] 缓存新程序 (当前缓存数: {len(self._program_cache)})")
                
                return result
            except Exception as e:
                # print(f"[FastPath] 快速编译失败: {e}, 回退到慢速路径")
                pass  # Fallback到慢速路径
        
        # 慢速路径: 完整AST求值
        fz = tx = ty = tz = 0.0
        try:
            for rule in program or []:
                cond = float(self._ast_eval(rule.get('condition'), state))
                if cond > 0.0:
                    for a in rule.get('action', []) or []:
                        try:
                            if hasattr(a, 'op') and a.op == 'set' and hasattr(a, 'left') and hasattr(a.left, 'value'):
                                key = str(getattr(a.left, 'value', ''))
                                val = float(self._ast_eval(getattr(a, 'right', 0.0), state))
                                if key == 'u_fz':
                                    fz += val
                                elif key == 'u_tx':
                                    tx += val
                                elif key == 'u_ty':
                                    ty += val
                                elif key == 'u_tz':
                                    tz += val
                        except Exception:
                            continue
        except Exception:
            pass
        # 适度裁剪（物理合理范围，经验值）
        fz = float(max(-5.0, min(5.0, fz)))     # N（向上为正）
        tx = float(max(-0.02, min(0.02, tx)))   # N*m
        ty = float(max(-0.02, min(0.02, ty)))   # N*m
        tz = float(max(-0.01, min(0.01, tz)))   # N*m（气动力矩较小）
        return fz, tx, ty, tz

    def _rpm_to_forces_local(self, rpm: np.ndarray) -> Tuple[float, float, float, float]:
        """将 4 电机 RPM 转换为 (fz, tx, ty, tz)，系数需与环境一致。"""
        KF = 2.8e-08
        KM = 1.1e-10
        L = 0.046
        omega = np.asarray(rpm, dtype=np.float64) * (2.0 * np.pi / 60.0)
        T = KF * (omega ** 2)
        fz = float(np.sum(T))
        tx = float(L * (T[1] - T[3]))
        ty = float(L * (T[2] - T[0]))
        tz = float(KM * (omega[0] ** 2 - omega[1] ** 2 + omega[2] ** 2 - omega[3] ** 2))
        return fz, tx, ty, tz

    def _target_pos(self, t: float) -> np.ndarray:
        """根据 trajectory_config 计算期望位置 [x,y,z]"""
        cfg = self.trajectory_config or {}
        tp = cfg.get('type', 'figure8')
        init = np.array(cfg.get('initial_xyz', [0.0, 0.0, 1.0]), dtype=np.float32)
        params = cfg.get('params', {})
        if tp == 'hover':
            # 悬停模式：目标点固定不动
            return init
        elif tp == 'circle':
            R = float(params.get('R', 0.9)); period = float(params.get('period', 10.0))
            w = 2.0 * np.pi / max(1e-6, period)
            x = R * np.cos(w * t); y = R * np.sin(w * t); z = 0.0
            return init + np.array([x, y, z], dtype=np.float32)
        elif tp == 'helix':
            R = float(params.get('R', 0.7)); period = float(params.get('period', 10.0)); vz = float(params.get('v_z', 0.15))
            w = 2.0 * np.pi / max(1e-6, period)
            x = R * np.cos(w * t); y = R * np.sin(w * t); z = vz * t
            return init + np.array([x, y, z], dtype=np.float32)
        else:  # figure8
            A = float(params.get('A', 0.8)); B = float(params.get('B', 0.5)); period = float(params.get('period', 12.0))
            w = 2.0 * np.pi / max(1e-6, period)
            x = A * np.sin(w * t)
            y = B * np.sin(w * t) * np.cos(w * t)
            z = 0.0
            return init + np.array([x, y, z], dtype=np.float32)
    
    def evaluate_batch(self, programs: List[List[Dict[str, Any]]]) -> List[float]:
        """
        使用Isaac Gym批量评估程序
        
        Args:
            programs: 程序列表，每个程序是规则列表
        
        Returns:
            rewards: 每个程序的奖励（负值=误差，越大越好）
        """
        # 初始化环境池
        if self._isaac_env_pool is None:
            self._init_isaac_gym_pool()

        # 延迟导入 torch：确保在 isaacgym 成功导入之后
        import torch  # type: ignore

        num_programs_original = len(programs)
        
        # 🔧 扩展replicas: 每个程序复制 replicas_per_program 次
        if self.replicas_per_program > 1:
            programs_expanded = []
            for prog in programs:
                programs_expanded.extend([prog] * self.replicas_per_program)
            programs = programs_expanded
        
        num_programs = len(programs)
        rewards = []
        
        start_time = time.time()
        
        # 分批评估（考虑replicas: 每批最多 isaac_num_envs // replicas_per_program 个程序）
        programs_per_batch = max(1, self.isaac_num_envs // self.replicas_per_program)
        
        for batch_start in range(0, num_programs, programs_per_batch):
            batch_end = min(batch_start + programs_per_batch, num_programs)
            batch_programs = programs[batch_start:batch_end]
            batch_size = len(batch_programs)
            
            # 🚀 环境池持久化优化: 只在必要时reset
            # 条件: 1) 首次使用 或 2) 需要更多环境数
            num_needed = batch_size
            should_reset = (not self._envs_ready) or (num_needed > self._last_reset_size)
            
            if should_reset:
                obs = self._isaac_env_pool.reset()
                self._envs_ready = True
                self._last_reset_size = self.isaac_num_envs
                if os.getenv('DEBUG_ENV_POOL', '0') == '1':
                    print(f"[BatchEvaluator] 🔄 Reset环境池 (需要{num_needed}个环境)")
            else:
                # 复用环境状态,直接获取观测 (避免7秒GPU同步开销!)
                obs = self._isaac_env_pool.get_obs()
                if os.getenv('DEBUG_ENV_POOL', '0') == '1':
                    print(f"[BatchEvaluator] ♻️ 复用环境池 (需要{num_needed}个,已有{self._last_reset_size}个) ⚡")
            
            # 运行仿真（环境池大小可能大于本批大小，按前 batch_size 个槽位使用）
            total_rewards = torch.zeros(self.isaac_num_envs, device=self.device)
            done_flags = torch.zeros(self.isaac_num_envs, dtype=torch.bool, device=self.device)
            # 为当前批次创建专属 done 标志和 stepwise 奖励计算器（匹配 batch_size）
            done_flags_batch = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            if self._step_reward_calc is not None:
                try:
                    weights, ks = get_reward_profile(self.reward_profile) if get_reward_profile else ({}, {})
                    self._step_reward_calc = StepwiseRewardCalculator(weights, ks, dt=self._step_dt, num_envs=batch_size, device=self.device)
                except Exception:
                    self._step_reward_calc = None
            # 记录每个环境累计了多少个有效步（用于 mean 归约）
            steps_count = torch.zeros(self.isaac_num_envs, device=self.device)
            # 记录是否曾经产生过非零动作（仅针对前 batch_size）
            ever_nonzero = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            
            # 初始化积分状态（持久化跨步）
            integral_states = [
                {
                    'err_i_x': 0.0, 'err_i_y': 0.0, 'err_i_z': 0.0,
                    'err_i_roll': 0.0, 'err_i_pitch': 0.0, 'err_i_yaw': 0.0
                }
                for i in range(batch_size)
            ]

            # 调试开关（需尽早声明，避免未定义引用）
            debug_enabled = bool(int(os.getenv('DEBUG_STEPWISE', '0')))

            # 准备每个程序对应的控制器/模式
            controllers = []
            use_u_flags = []  # True 表示该程序直接输出 (fz,tx,ty,tz)
            try:
                from .segmented_controller import PiLightSegmentedPIDController
            except ImportError:
                try:
                    from segmented_controller import PiLightSegmentedPIDController
                except ImportError:
                    PiLightSegmentedPIDController = None  # type: ignore
            if self.strict_no_prior:
                # 严格无先验：统一走 u_* 路径
                controllers = [None for _ in range(batch_size)]
                use_u_flags = [True for _ in range(batch_size)]
                if debug_enabled:
                    print("[DebugReward] strict_no_prior=ON → all programs use direct u_* path")
            else:
                if PiLightSegmentedPIDController is not None:
                    for prog in batch_programs:
                        if self._program_uses_u(prog):
                            controllers.append(None)
                            use_u_flags.append(True)
                        else:
                            controllers.append(
                                PiLightSegmentedPIDController(
                                    program=prog,
                                    suppress_init_print=True,
                                    semantics='compose_by_gain',
                                    min_hold_steps=2
                                )
                            )
                            use_u_flags.append(False)
                    # 调试：统计本批可解析的分段规则数量
                    if debug_enabled:
                        try:
                            seg_counts = []
                            for i in range(len(controllers)):
                                if controllers[i] is None:
                                    seg_counts.append(-1)  # -1 表示走 u_* 路径
                                else:
                                    try:
                                        seg_counts.append(int(len(getattr(controllers[i], 'segments', []) or [])))
                                    except Exception:
                                        seg_counts.append(0)
                            print("[DebugReward] controller segments per-prog:", seg_counts[:min(8, len(seg_counts))])
                        except Exception:
                            pass
                else:
                    controllers = [None for _ in range(batch_size)]
                    # 无控制器实现时，一律走 u_* 路径（若程序不含 u_*，则保持 0）
                    for prog in batch_programs:
                        use_u_flags.append(self._program_uses_u(prog))

            # 控制步数（以控制频率计，不再按物理频率）
            max_steps = int(self.duration * float(getattr(self, '_control_freq', 48)))
            min_steps = int(max_steps * self.min_steps_frac)
            
            # 调试辅助：记录首末位置误差（仅在开启 DEBUG_STEPWISE 时）
            first_pos_err = None
            last_pos_err = None

            for step in range(max_steps):
                # 计算目标点（所有 env 相同目标轨迹，使用动态轨迹而不是静态 cfg.target）
                t = step * float(getattr(self, '_control_dt', 1.0/48.0))
                tgt_np = self._target_pos(t)  # numpy array [3]
                tgt_tensor = torch.tensor(tgt_np, device=self.device, dtype=torch.float32)

                # 生成动作（统一为 [fx,fy,fz,tx,ty,tz] 6 维格式，便于混用）
                actions = torch.zeros((self.isaac_num_envs, 6), device=self.device)
                pos = obs['position'][:batch_size]
                quat = obs['orientation'][:batch_size]
                vel = obs['velocity'][:batch_size]
                omega = obs['angular_velocity'][:batch_size]
                
                # 🚀🚀 超高性能路径: 完全向量化 + JIT
                if self.use_fast_path and self._ultra_executor is not None and step == 0:
                    # 首次步骤: 预编译所有程序 (只做一次)
                    try:
                        if not hasattr(self, '_compiled_forces'):
                            self._compiled_forces = self._ultra_executor.compile_programs(batch_programs)
                            print(f"[UltraFast] ✅ 预编译{len(batch_programs)}程序 → 缓存{len(self._ultra_executor.program_cache)}个唯一程序")
                    except Exception as e:
                        print(f"[UltraFast] ⚠️ 预编译失败: {e}, 回退到标准快速路径")
                        self._ultra_executor = None
                
                # 🚀 快速路径: 批量处理 u_* 路径
                if self.use_fast_path:
                    # 预先导入scipy（避免循环内重复导入）
                    try:
                        from scipy.spatial.transform import Rotation
                    except ImportError:
                        Rotation = None
                    
                    # 批量计算位置误差 [batch_size, 3]
                    # 注意: Isaac Gym的obs可能是torch tensor或numpy array
                    if isinstance(pos, torch.Tensor):
                        pos_np = pos.cpu().numpy()
                        quat_np = quat.cpu().numpy()
                        vel_np = vel.cpu().numpy()
                        omega_np = omega.cpu().numpy()
                    else:
                        pos_np = np.asarray(pos)
                        quat_np = np.asarray(quat)
                        vel_np = np.asarray(vel)
                        omega_np = np.asarray(omega)
                    
                    tgt_batch = np.tile(tgt_np, (batch_size, 1))  # [batch_size, 3]
                    pe_batch = tgt_batch - pos_np  # [batch_size, 3]
                    
                    # 批量计算RPY
                    if Rotation is not None:
                        try:
                            rpy_batch = Rotation.from_quat(quat_np).as_euler('XYZ', degrees=False)  # [batch_size, 3]
                        except Exception:
                            rpy_batch = np.zeros((batch_size, 3), dtype=np.float32)
                    else:
                        rpy_batch = np.zeros((batch_size, 3), dtype=np.float32)
                    
                    # 🚀🚀 超高性能执行: 批量应用预编译的力
                    if self._ultra_executor is not None and hasattr(self, '_compiled_forces'):
                        try:
                            # 批量执行 (消除Python循环)
                            try:
                                from .ultra_fast_executor import apply_forces_jit, update_integral_jit
                            except ImportError:
                                from ultra_fast_executor import apply_forces_jit, update_integral_jit
                            
                            use_u_array = np.array(use_u_flags, dtype=np.bool_)
                            actions_np = np.zeros((batch_size, 6), dtype=np.float32)
                            apply_forces_jit(actions_np, self._compiled_forces, use_u_array)
                            
                            # 转为tensor
                            actions[:batch_size] = torch.from_numpy(actions_np).to(self.device)
                            
                            # 更新积分项 (JIT加速)
                            if not all(done_flags[:batch_size].cpu().numpy()):
                                err_i = np.array([
                                    [s['err_i_x'], s['err_i_y'], s['err_i_z'],
                                     s['err_i_roll'], s['err_i_pitch'], s['err_i_yaw']]
                                    for s in integral_states
                                ], dtype=np.float32)
                                done_array = done_flags[:batch_size].cpu().numpy().astype(np.bool_)
                                dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                update_integral_jit(err_i, pe_batch, rpy_batch, done_array, dt)
                                
                                # 写回integral_states
                                for i in range(batch_size):
                                    integral_states[i]['err_i_x'] = float(err_i[i, 0])
                                    integral_states[i]['err_i_y'] = float(err_i[i, 1])
                                    integral_states[i]['err_i_z'] = float(err_i[i, 2])
                                    integral_states[i]['err_i_roll'] = float(err_i[i, 3])
                                    integral_states[i]['err_i_pitch'] = float(err_i[i, 4])
                                    integral_states[i]['err_i_yaw'] = float(err_i[i, 5])
                            
                            # 检查ever_nonzero (向量化)
                            if self.strict_no_prior:
                                nonzero_mask = (np.abs(actions_np[:, 2]) > 1e-6) | \
                                               (np.abs(actions_np[:, 3]) > 1e-8) | \
                                               (np.abs(actions_np[:, 4]) > 1e-8) | \
                                               (np.abs(actions_np[:, 5]) > 1e-8)
                                for i in range(batch_size):
                                    if use_u_flags[i] and nonzero_mask[i]:
                                        ever_nonzero[i] = True
                            
                            # 处理非u_*路径（PID控制器）
                            for i in range(batch_size):
                                if not use_u_flags[i]:
                                    ctrl = controllers[i]
                                    try:
                                        if ctrl is not None:
                                            pe = pe_batch[i]
                                            ctrl_actions = ctrl.step(
                                                time_step=step,
                                                pos_x=float(pos[i][0]),
                                                pos_y=float(pos[i][1]),
                                                pos_z=float(pos[i][2]),
                                                target_x=float(tgt_np[0]),
                                                target_y=float(tgt_np[1]),
                                                target_z=float(tgt_np[2]),
                                            )
                                            actions[i, 0] = float(ctrl_actions.get('fx', 0.0))
                                            actions[i, 1] = float(ctrl_actions.get('fy', 0.0))
                                            actions[i, 2] = float(ctrl_actions.get('fz', 0.0))
                                            actions[i, 3] = float(ctrl_actions.get('tx', 0.0))
                                            actions[i, 4] = float(ctrl_actions.get('ty', 0.0))
                                            actions[i, 5] = float(ctrl_actions.get('tz', 0.0))
                                            if self.strict_no_prior:
                                                if (abs(actions[i, 2]) > 1e-6) or (abs(actions[i, 3]) > 1e-8) or \
                                                   (abs(actions[i, 4]) > 1e-8) or (abs(actions[i, 5]) > 1e-8):
                                                    ever_nonzero[i] = True
                                            
                                            # 更新积分项
                                            dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                            integral_states[i]['err_i_x'] += pe[0] * dt
                                            integral_states[i]['err_i_y'] += pe[1] * dt
                                            integral_states[i]['err_i_z'] += pe[2] * dt
                                    except Exception as e:
                                        if debug_enabled:
                                            print(f"[DebugReward] Controller step failed for env {i}: {e}")
                                        pass
                            
                        except Exception as e:
                            if step == 0:
                                import traceback
                                print(f"[UltraFast] ⚠️ 执行失败: {e}")
                                traceback.print_exc()
                            print(f"[UltraFast] 回退到标准路径")
                            # 回退到下面的标准快速路径
                            self._ultra_executor = None
                    
                    # 标准快速路径 (如果超高性能路径未激活)
                    if self._ultra_executor is None or not hasattr(self, '_compiled_forces'):
                        # 向量化处理所有使用u_*的程序
                        for i in range(batch_size):
                            if use_u_flags[i]:
                                pe = pe_batch[i]
                                rpy = rpy_batch[i]
                                
                                state = {
                                'pos_err_x': float(pe[0]),
                                'pos_err_y': float(pe[1]),
                                'pos_err_z': float(pe[2]),
                                'pos_err': float(np.linalg.norm(pe)),
                                'pos_err_xy': float(np.linalg.norm(pe[:2])),
                                'pos_err_z_abs': float(abs(pe[2])),
                                'vel_x': float(vel_np[i][0]),
                                'vel_y': float(vel_np[i][1]),
                                'vel_z': float(vel_np[i][2]),
                                'vel_err': float(np.linalg.norm(vel_np[i])),
                                'err_p_roll': float(rpy[0]),
                                'err_p_pitch': float(rpy[1]),
                                'err_p_yaw': float(rpy[2]),
                                'ang_err': float(np.linalg.norm(rpy)),
                                'rpy_err_mag': float(np.linalg.norm(rpy)),
                                'ang_vel_x': float(omega_np[i][0]),
                                'ang_vel_y': float(omega_np[i][1]),
                                'ang_vel_z': float(omega_np[i][2]),
                                'ang_vel': float(np.linalg.norm(omega_np[i])),
                                'ang_vel_mag': float(np.linalg.norm(omega_np[i])),
                                'err_i_x': float(integral_states[i]['err_i_x']),
                                'err_i_y': float(integral_states[i]['err_i_y']),
                                'err_i_z': float(integral_states[i]['err_i_z']),
                                'err_i_roll': float(integral_states[i]['err_i_roll']),
                                'err_i_pitch': float(integral_states[i]['err_i_pitch']),
                                'err_i_yaw': float(integral_states[i]['err_i_yaw']),
                                'err_d_x': float(-vel_np[i][0]),
                                'err_d_y': float(-vel_np[i][1]),
                                'err_d_z': float(-vel_np[i][2]),
                                    'err_d_roll': float(-omega_np[i][0]),
                                    'err_d_pitch': float(-omega_np[i][1]),
                                    'err_d_yaw': float(-omega_np[i][2]),
                                }
                                fz, tx, ty, tz = self._eval_program_forces(batch_programs[i], state)
                                actions[i, 0] = 0.0
                                actions[i, 1] = 0.0
                                actions[i, 2] = float(fz)
                                actions[i, 3] = float(tx)
                                actions[i, 4] = float(ty)
                                actions[i, 5] = float(tz)
                                if self.strict_no_prior:
                                    if (abs(fz) > 1e-6) or (abs(tx) > 1e-8) or (abs(ty) > 1e-8) or (abs(tz) > 1e-8):
                                        ever_nonzero[i] = True
                                
                                # 更新积分项
                                dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                integral_states[i]['err_i_x'] += pe[0] * dt
                                integral_states[i]['err_i_y'] += pe[1] * dt
                                integral_states[i]['err_i_z'] += pe[2] * dt
                                integral_states[i]['err_i_roll'] += rpy[0] * dt
                                integral_states[i]['err_i_pitch'] += rpy[1] * dt
                                integral_states[i]['err_i_yaw'] += rpy[2] * dt
                    
                    # 处理非u_*路径（PID控制器）
                    for i in range(batch_size):
                        if not use_u_flags[i]:
                            ctrl = controllers[i]
                            try:
                                if ctrl is not None:
                                    pe = pe_batch[i]
                                    ctrl_actions = ctrl.step(
                                        time_step=step,
                                        pos_x=float(pos[i][0]),
                                        pos_y=float(pos[i][1]),
                                        pos_z=float(pos[i][2]),
                                        target_x=float(tgt_np[0]),
                                        target_y=float(tgt_np[1]),
                                        target_z=float(tgt_np[2]),
                                    )
                                    actions[i, 0] = float(ctrl_actions.get('fx', 0.0))
                                    actions[i, 1] = float(ctrl_actions.get('fy', 0.0))
                                    actions[i, 2] = float(ctrl_actions.get('fz', 0.0))
                                    actions[i, 3] = float(ctrl_actions.get('tx', 0.0))
                                    actions[i, 4] = float(ctrl_actions.get('ty', 0.0))
                                    actions[i, 5] = float(ctrl_actions.get('tz', 0.0))
                                    if self.strict_no_prior:
                                        if (abs(actions[i, 2]) > 1e-6) or (abs(actions[i, 3]) > 1e-8) or \
                                           (abs(actions[i, 4]) > 1e-8) or (abs(actions[i, 5]) > 1e-8):
                                            ever_nonzero[i] = True
                                    
                                    # 更新积分项
                                    dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                    integral_states[i]['err_i_x'] += pe[0] * dt
                                    integral_states[i]['err_i_y'] += pe[1] * dt
                                    integral_states[i]['err_i_z'] += pe[2] * dt
                            except Exception as e:
                                if debug_enabled:
                                    print(f"[DebugReward] Controller step failed for env {i}: {e}")
                                pass
                else:
                    # 慢速路径: 原始串行处理
                    for i in range(batch_size):
                        ctrl = controllers[i]
                        try:
                            if use_u_flags[i]:
                                # 构造完整三轴 state（支持精细 PID）
                                pe = np.asarray(tgt_np, dtype=np.float32) - np.asarray(pos[i], dtype=np.float32)
                                # 获取四元数 → RPY（简化：仅用于姿态误差估算）
                                try:
                                    from scipy.spatial.transform import Rotation
                                    rpy = Rotation.from_quat(quat[i]).as_euler('XYZ', degrees=False)
                                except Exception:
                                    # 无 scipy 时退化为零
                                    rpy = np.zeros(3, dtype=np.float32)
                                
                                # TODO: 积分项需要跨步累积（当前简化为零）
                                state = {
                                # 位置误差（三轴）
                                'pos_err_x': float(pe[0]),
                                'pos_err_y': float(pe[1]),
                                'pos_err_z': float(pe[2]),
                                'pos_err': float(np.linalg.norm(pe)),
                                'pos_err_xy': float(np.linalg.norm(pe[:2])),
                                'pos_err_z_abs': float(abs(pe[2])),
                                # 速度（三轴 + 模长）
                                'vel_x': float(vel[i][0]),
                                'vel_y': float(vel[i][1]),
                                'vel_z': float(vel[i][2]),
                                'vel_err': float(np.linalg.norm(vel[i])),
                                # 姿态误差（RPY，目标假设为 0）
                                'err_p_roll': float(rpy[0]),
                                'err_p_pitch': float(rpy[1]),
                                'err_p_yaw': float(rpy[2]),
                                'ang_err': float(np.linalg.norm(rpy)),
                                'rpy_err_mag': float(np.linalg.norm(rpy)),
                                # 角速度（三轴 + 模长）
                                'ang_vel_x': float(omega[i][0]),
                                'ang_vel_y': float(omega[i][1]),
                                'ang_vel_z': float(omega[i][2]),
                                'ang_vel': float(np.linalg.norm(omega[i])),
                                'ang_vel_mag': float(np.linalg.norm(omega[i])),
                                # 积分项（累积）
                                'err_i_x': float(integral_states[i]['err_i_x']),
                                'err_i_y': float(integral_states[i]['err_i_y']),
                                'err_i_z': float(integral_states[i]['err_i_z']),
                                'err_i_roll': float(integral_states[i]['err_i_roll']),
                                'err_i_pitch': float(integral_states[i]['err_i_pitch']),
                                'err_i_yaw': float(integral_states[i]['err_i_yaw']),
                                # 微分项（近似为速度/角速度的负值）
                                'err_d_x': float(-vel[i][0]),
                                'err_d_y': float(-vel[i][1]),
                                'err_d_z': float(-vel[i][2]),
                                'err_d_roll': float(-omega[i][0]),
                                'err_d_pitch': float(-omega[i][1]),
                                'err_d_yaw': float(-omega[i][2]),
                                }
                                fz, tx, ty, tz = self._eval_program_forces(batch_programs[i], state)
                                actions[i, 0] = 0.0
                                actions[i, 1] = 0.0
                                actions[i, 2] = float(fz)
                                actions[i, 3] = float(tx)
                                actions[i, 4] = float(ty)
                                actions[i, 5] = float(tz)
                                # 记录是否产生非零动作
                                if self.strict_no_prior:
                                    if (abs(fz) > 1e-6) or (abs(tx) > 1e-8) or (abs(ty) > 1e-8) or (abs(tz) > 1e-8):
                                        ever_nonzero[i] = True
                                # 更新积分状态（仅对未完成的环境）
                                if not done_flags[i]:
                                    dt = float(self._control_dt)
                                    integral_states[i]['err_i_x'] += float(pe[0]) * dt
                                    integral_states[i]['err_i_y'] += float(pe[1]) * dt
                                    integral_states[i]['err_i_z'] += float(pe[2]) * dt
                                    integral_states[i]['err_i_roll'] += float(rpy[0]) * dt
                                    integral_states[i]['err_i_pitch'] += float(rpy[1]) * dt
                                    integral_states[i]['err_i_yaw'] += float(rpy[2]) * dt
                            else:
                                if ctrl is None:
                                    continue
                                rpm, _pos_e, _rpy_e = ctrl.computeControl(
                                    self._control_dt,
                                    cur_pos=pos[i],
                                    cur_quat=quat[i],
                                    cur_vel=vel[i],
                                    cur_ang_vel=omega[i],
                                    target_pos=tgt_np,
                                )
                                rpm = np.clip(np.asarray(rpm, dtype=np.float32), 0.0, 25000.0)
                                fz, tx, ty, tz = self._rpm_to_forces_local(rpm)
                                actions[i, 2] = float(fz)
                                actions[i, 3] = float(tx)
                                actions[i, 4] = float(ty)
                                actions[i, 5] = float(tz)
                        except Exception:
                            # 失败则保持零动作
                            pass
                
                # 步进仿真
                obs, step_rewards_env, dones, infos = self._isaac_env_pool.step(actions)

                # 自定义奖励：轨迹跟踪 + 速度惩罚 + 控制能量惩罚 + 坠毁惩罚
                import torch
                pos = torch.tensor(obs['position'], device=self.device, dtype=torch.float32)
                vel = torch.tensor(obs['velocity'], device=self.device, dtype=torch.float32)
                omega = torch.tensor(obs['angular_velocity'], device=self.device, dtype=torch.float32)
                # 目标（悬停或轨迹）
                if self.trajectory_config.get('type') == 'hover':
                    tgt = np.array([0.0, 0.0, self.trajectory_config.get('height', 1.0)], dtype=np.float32)
                else:
                    tgt = np.array(self.trajectory_config.get('target', [0.0, 0.0, 1.0]), dtype=np.float32)
                # Stepwise 奖励
                if self._step_reward_calc is not None:
                    step_total = self._step_reward_calc.compute_step(
                        pos[:batch_size, :],
                        tgt_tensor,
                        vel[:batch_size, :],
                        omega[:batch_size, :],
                        actions[:batch_size, :],
                        done_flags_batch
                    )
                    step_reward = step_total
                else:
                    # 退回旧逻辑
                    # 悬停模式：加大位置权重，降低速度容忍度
                    if self.trajectory_config.get('type') == 'hover':
                        w_pos, w_vel = 2.0, 0.3  # 悬停：更看重精确定点和静止
                    else:
                        w_pos, w_vel = 1.0, 0.1  # 轨迹跟踪：允许一定速度
                    pos_err = pos[:batch_size, :] - tgt_tensor
                    step_reward = - w_pos * torch.norm(pos_err, dim=1)
                    step_reward -= w_vel * torch.norm(vel[:batch_size, :], dim=1)
                    act_pen = 1e-7 * torch.sum(actions[:batch_size, :] ** 2, dim=1)
                    step_reward -= act_pen
                    crashed = pos[:batch_size, 2] < 0.1
                    step_reward[crashed] -= 5.0

                # 调试：记录首末位置误差（使用动态目标）
                if debug_enabled:
                    # 计算当前步的绝对位置误差模长
                    cur_pos_err = torch.norm(pos[:batch_size, :] - tgt_tensor.view(1, 3), dim=1)
                    if step == 0:
                        first_pos_err = cur_pos_err.detach()[:min(8, batch_size)].cpu()
                    last_pos_err = cur_pos_err.detach()[:min(8, batch_size)].cpu()
                # 累积奖励
                active_mask = (~done_flags_batch).float()
                total_rewards[:batch_size] += step_reward * active_mask
                steps_count[:batch_size] += active_mask
                # 更新批次 done 标志（仅前 batch_size 有效）
                done_flags_batch |= dones[:batch_size]
                done_flags[:batch_size] = done_flags_batch
                if step >= min_steps and done_flags_batch.all():
                    break
            # 额外的 episode 末尾奖励
            if self._step_reward_calc is not None:
                bonus = self._step_reward_calc.finalize()[:batch_size]
                total_rewards[:batch_size] += bonus
            # 在严格无先验模式下：对整集始终零动作的程序施加惩罚
            if self.strict_no_prior and self.zero_action_penalty > 0:
                zero_mask = (~ever_nonzero).float()
                total_rewards[:batch_size] -= self.zero_action_penalty * zero_mask
                if debug_enabled:
                    try:
                        zero_cnt = int((~ever_nonzero).sum().item())
                        print(f"[DebugReward] zero-action programs in batch: {zero_cnt}/{batch_size}")
                    except Exception:
                        pass
            # 归约
            if self.reward_reduction == 'mean':
                denom = torch.clamp(steps_count[:batch_size], min=1.0)
                batch_scores = (total_rewards[:batch_size] / denom).cpu().numpy().tolist()
            else:
                batch_scores = total_rewards[:batch_size].cpu().numpy().tolist()
            rewards.extend(batch_scores)

            # 调试输出（仅首批 & 开启时）
            if debug_enabled and batch_start == 0:
                try:
                    print("[DebugReward] batch_size={} mean_final_reward={:.4f}".format(
                        batch_size, float(np.mean(batch_scores))))
                    if first_pos_err is not None and last_pos_err is not None:
                        diff = (last_pos_err - first_pos_err).numpy()
                        print("[DebugReward] first_pos_err[:8] =", [f"{x:.3f}" for x in first_pos_err.numpy()])
                        print("[DebugReward] last_pos_err[:8]  =", [f"{x:.3f}" for x in last_pos_err.numpy()])
                        print("[DebugReward] Δpos_err[:8]      =", [f"{x:.3f}" for x in diff])
                except Exception:
                    pass
        
        elapsed = time.time() - start_time
        # 显示原始程序数(未扩展replicas前)
        display_count = num_programs_original if self.replicas_per_program > 1 else num_programs
        print(f"[BatchEvaluator] ✅ 评估完成: {display_count} 程序 (×{self.replicas_per_program} replicas), {elapsed:.2f}秒 ({elapsed/display_count*1000:.1f}ms/程序)")
        
        # 🔧 如果使用了replicas, 对每个原始程序的replicas求平均
        if self.replicas_per_program > 1:
            averaged_rewards = []
            for i in range(num_programs_original):
                start_idx = i * self.replicas_per_program
                end_idx = start_idx + self.replicas_per_program
                avg_reward = float(np.mean(rewards[start_idx:end_idx]))
                averaged_rewards.append(avg_reward)
            return averaged_rewards
        
        return rewards
    
    def _compute_action_from_program(self, program: List[Dict[str, Any]], 
                                      obs: np.ndarray, step: int) -> np.ndarray:
        """
        从程序计算控制输入（简化版）
        
        Args:
            program: DSL程序规则列表
            obs: 观测 [obs_dim]
            step: 当前步数
        
        Returns:
            action: [4] = [thrust, roll_rate, pitch_rate, yaw_rate]
        
        TODO: 集成完整的 PiLightSegmentedPIDController
        """
        # 当前返回悬停控制（占位符）
        # 实际应该：
        # 1. 从obs提取状态（位置、速度等）
        # 2. 计算轨迹目标点
        # 3. 使用program规则计算PID输出
        # 4. 转换为电机指令
        
        return np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def evaluate_single(self, program: List[Dict[str, Any]]) -> float:
        """评估单个程序：可并行复制多个副本并取平均，提升GPU利用率/稳定性"""
        if self.replicas_per_program <= 1:
            return self.evaluate_batch([program])[0]
        else:
            programs = [program] * self.replicas_per_program
            rewards = self.evaluate_batch(programs)
            return float(np.mean(rewards))


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试Isaac Gym批量评估器")
    print("=" * 80)
    
    if not ISAAC_GYM_AVAILABLE:
        print("❌ Isaac Gym未安装，无法测试")
        exit(1)
    
    trajectory = {
        'type': 'figure8',
        'initial_xyz': [0, 0, 1.0],
        'params': {'A': 0.8, 'B': 0.5, 'period': 12}
    }
    
    evaluator = BatchEvaluator(
        trajectory_config=trajectory,
        duration=5,
        isaac_num_envs=64,
        device='cuda:0'
    )
    
    # 创建测试程序
    test_programs = [
        [{'name': 'rule1', 'condition': None, 'action': [], 'multiplier': [1, 1, 1]}]
    ] * 8
    
    print(f"\n评估 {len(test_programs)} 个程序...")
    rewards = evaluator.evaluate_batch(test_programs)
    print(f"奖励: {[f'{r:.3f}' for r in rewards]}")
    print("\n✅ 测试完成")
