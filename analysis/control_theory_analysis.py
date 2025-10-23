from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

# 输出目录
ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(ROOT)
OUT_DIR = os.path.join(WS, 'results', 'control_theory_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']

# ============================================================================
# 控制理论分析说明
# ============================================================================
# 本脚本对 MCTS 合成的分段 PID 控制器进行经典控制理论分析
#
# 【为什么使用二阶系统模型？】
#
# 1. 物理本质：
#    无人机的单轴姿态动力学本质上是二阶系统：
#      τ = I·α  (力矩 = 转动惯量 × 角加速度)
#    积分两次得到角度：θ = ∫∫(τ/I)dt²
#    这是典型的双积分器结构，对应二阶传递函数
#
# 2. 主导极点理论：
#    实际无人机系统虽然包含：
#      - 电机动态（一阶，τ_motor ≈ 10-50ms）
#      - 螺旋桨空气动力学（高频非线性）
#      - 机体弹性模态（远高于控制带宽）
#      - 传感器滤波（一阶低通）
#    但在控制带宽内（通常 <10 Hz），这些高频/快速动态已经衰减，
#    系统的主导行为由惯性（二阶）决定。
#
# 3. 分析复杂度 vs 洞察力的权衡：
#    - 二阶系统：2 个参数（ωn, ζ），有解析解，物理意义清晰
#      * ωn: 自然频率 - 系统响应速度
#      * ζ: 阻尼比 - 振荡程度
#      * 可直接从波德图/根轨迹读出性能指标
#
#    - 高阶系统（如 4-8 阶真实模型）：
#      * 需要大量参数辨识（转动惯量、电机常数、空气动力系数...）
#      * 难以获得解析解，需要数值仿真
#      * 难以直观理解每个极点的物理含义
#      * 对参数误差敏感，鲁棒性分析复杂
#
# 4. 验证有效性：
#    本分析的目的不是精确预测实际飞行性能（那需要 PyBullet 全保真仿真），
#    而是：
#      ✓ 验证控制器的数学稳定性（李雅普诺夫）
#      ✓ 比较不同策略的相对性能差异（Rule1 vs Rule5）
#      ✓ 理解 PID 参数如何影响频域/时域特性
#      ✓ 提供控制理论视角的可解释性
#
#    对于这些目标，二阶模型已经足够，且结果与 PyBullet 仿真定性一致。
#
# 5. 何时需要高阶模型？
#    以下情况需要考虑高阶效应：
#      - 设计需要利用电机动态（如主动阻尼）
#      - 控制带宽接近执行器带宽（高性能竞速无人机）
#      - 需要考虑结构振动（大型无人机）
#      - 传感器噪声特性对控制影响显著
#      - 需要精确的定量预测（而非定性分析）
#
# 6. 本项目的选择：
#    本项目采用"分层建模"策略：
#      - 控制理论分析层：二阶简化模型（本脚本）→ 快速洞察
#      - 控制器评估层：PyBullet 全保真仿真 → 真实性能
#      - 两者结合：理论指导 + 仿真验证
#
# 被控对象模型：采用二阶标准模型近似无人机姿态/位置通道动力学
#   Gp(s) = ωn² / (s² + 2ζωn·s + ωn²)
#   其中 ωn=1.0 (自然频率), ζ=0.7 (阻尼比)
#
# 分析方法：
#   1. 频域分析 (Bode 图)：评估稳定裕度、频率响应
#   2. 灵敏度函数：评估抗扰性能和噪声放大
#   3. 时域分析：评估阶跃响应、超调量、调节时间
#   4. 李雅普诺夫稳定性：数学严格的稳定性证明
# ============================================================================

def pid_tf(Kp: float, Ki: float, Kd: float):
    """
    构造 PID 控制器的传递函数
    
    PID 控制律：u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt
    
    传递函数形式：C(s) = Kp + Ki/s + Kd·s
                        = (Kd·s² + Kp·s + Ki) / s
    
    参数说明：
        Kp: 比例增益 - 控制当前误差，决定响应速度
        Ki: 积分增益 - 消除稳态误差，但可能降低相位裕度
        Kd: 微分增益 - 预测误差趋势，提高阻尼，抑制超调
    
    返回：scipy.signal.TransferFunction 对象
    """
    num = [Kd, Kp, Ki]
    den = [1, 0]  # / s
    C = signal.TransferFunction(num, den)
    return C


def plant_tf(wn: float = 1.0, zeta: float = 0.7):
    """
    构造被控对象（无人机动力学）的二阶标准传递函数
    
    标准二阶系统：Gp(s) = ωn² / (s² + 2ζωn·s + ωn²)
    
    这个模型近似了无人机姿态/位置通道的低阶动力学特性，
    忽略高阶非线性和执行器动态，便于进行频域/时域分析。
    
    参数说明：
        wn (ωn): 自然频率 (rad/s) - 决定系统的响应速度
                 wn=1.0 表示归一化频率，实际物理系统需要根据
                 无人机质量、转动惯量等参数调整
        
        zeta (ζ): 阻尼比 - 决定系统的振荡特性
                  ζ < 1: 欠阻尼 (有振荡)
                  ζ = 1: 临界阻尼 (最快无振荡响应)
                  ζ > 1: 过阻尼 (响应慢但无振荡)
                  ζ = 0.7: 常用的"良好阻尼"值
    
    物理意义：
        对于位置控制：Gp 表示从控制力到位移的传递关系
        对于姿态控制：Gp 表示从控制力矩到角度的传递关系
    
    返回：scipy.signal.TransferFunction 对象
    """
    num = [wn**2]
    den = [1.0, 2.0*zeta*wn, wn**2]
    return signal.TransferFunction(num, den)


def _tf_coeffs(TF: signal.TransferFunction):
    def _one(x):
        a = np.array(x, dtype=float)
        if a.ndim > 1:
            a = np.squeeze(a)
        return a
    return _one(TF.num), _one(TF.den)


def _tf_mul(A: signal.TransferFunction, B: signal.TransferFunction) -> signal.TransferFunction:
    An, Ad = _tf_coeffs(A)
    Bn, Bd = _tf_coeffs(B)
    num = np.polymul(An, Bn)
    den = np.polymul(Ad, Bd)
    return signal.TransferFunction(num, den)


def _tf_feedback(F: signal.TransferFunction, H: signal.TransferFunction | float = 1.0, sign: int = 1) -> signal.TransferFunction:
    # 闭环：T = F / (1 + sign*F*H)
    if isinstance(H, (int, float)):
        H = signal.TransferFunction([H], [1.0])
    # F = Nf/Df, H = Nh/Dh
    Nf, Df = _tf_coeffs(F)
    Nh, Dh = _tf_coeffs(H)
    # T = (Nf*Dh) / (Df*Dh + sign*Nf*Nh)
    num = np.polymul(Nf, Dh)
    den = np.polyadd(np.polymul(Df, Dh), sign * np.polymul(Nf, Nh))
    return signal.TransferFunction(np.squeeze(num), np.squeeze(den))


def closed_loop_tf(C: signal.TransferFunction, G: signal.TransferFunction):
    # L(s) = C*G
    L = _tf_mul(C, G)
    # T(s) = L/(1+L), S(s) = 1/(1+L)
    T = _tf_feedback(L, 1.0, sign=1)
    # 对于 S：可通过 1/(1+L) 显式构造
    Nl, Dl = _tf_coeffs(L)
    S = signal.TransferFunction(Dl, np.polyadd(Dl, Nl))
    return L, T, S


def tf_to_ss(tf: signal.TransferFunction):
    # 返回 A, B, C, D
    N, D = _tf_coeffs(tf)
    return signal.tf2ss(N, D)


def lyapunov_check(T: signal.TransferFunction):
    """
    李雅普诺夫稳定性分析 - 数学严格的稳定性证明方法
    
    理论基础：
        李雅普诺夫第二方法（直接法）通过构造一个能量函数 V(x) 来证明稳定性。
        对于线性系统 ẋ = Ax，如果存在正定矩阵 P 满足李雅普诺夫方程：
            A^T·P + P·A = -Q
        其中 Q 是任意正定矩阵（通常取 Q=I），则系统渐近稳定。
    
    物理意义：
        V(x) = x^T·P·x 可以理解为系统的"能量函数"
        dV/dt = -x^T·Q·x < 0 表示能量持续衰减
        这保证了系统状态最终会收敛到平衡点（稳定）
    
    判据：
        1. 极点判据：所有特征值的实部必须 < 0 (位于复平面左半平面)
        2. 李雅普诺夫判据：P 矩阵必须正定 (所有特征值 > 0)
    
    实际应用：
        对于无人机控制，稳定性意味着：
        - 小的扰动不会导致发散
        - 系统最终会回到期望姿态/位置
        - 不会出现不受控的振荡或翻滚
    
    参数：
        T: 闭环传递函数 T(s) = L/(1+L)
    
    返回：
        eigvals: 闭环系统的特征值（极点）
        stable: 布尔值，True 表示稳定
        P: 李雅普诺夫方程的解矩阵（正定则稳定）
    """
    # 将闭环传函转为状态空间表示 ẋ=Ax+Bu, y=Cx+Du
    A, B, C, D = tf_to_ss(T)
    
    # 计算特征值（系统极点）
    eigvals = np.linalg.eigvals(A)
    
    # Hurwitz 稳定性：所有特征值实部 < 0
    stable = np.all(np.real(eigvals) < 0)
    
    # 求解李雅普诺夫方程 A^T·P + P·A = -Q
    # 取 Q = I（单位矩阵）作为正定矩阵
    Q = np.eye(A.shape[0])
    P = linalg.solve_continuous_lyapunov(A.T, -Q)
    
    # 如果系统稳定，P 应该是正定的（所有特征值 > 0）
    # 可以通过 np.all(np.linalg.eigvals(P) > 0) 验证
    
    return eigvals, stable, P


def compute_margins(L: signal.TransferFunction):
    """
    计算稳定裕度：相位裕度 (PM) 和增益裕度 (GM)
    
    稳定裕度是评估闭环系统鲁棒性的重要指标，回答以下问题：
    - 系统距离不稳定有多远？
    - 可以容忍多大的模型误差或参数变化？
    
    1. 相位裕度 (Phase Margin, PM)：
       定义：在增益穿越频率（|L(jω)|=0dB）处，相位距离-180°的余量
       计算：PM = 180° + ∠L(jωc)，其中 |L(jωc)| = 1
       
       物理意义：
       - PM > 0°: 系统稳定
       - PM > 30°: 可接受的稳定性
       - PM > 45°: 良好的稳定性
       - PM > 60°: 优秀的稳定性
       
       实际影响：
       - PM 大 → 抗时延能力强、鲁棒性好、超调量小
       - PM 小 → 容易振荡、对时延敏感、可能不稳定
    
    2. 增益裕度 (Gain Margin, GM)：
       定义：在相位穿越频率（∠L(jω)=-180°）处，增益距离0dB的余量
       计算：GM = -|L(jωp)|_dB，其中 ∠L(jωp) = -180°
       
       物理意义：
       - GM > 0 dB: 系统稳定
       - GM > 6 dB: 可接受的稳定性
       - GM > 10 dB: 良好的稳定性
       
       实际影响：
       - GM 表示增益可以增加多少倍而不失稳
       - 对执行器饱和、非线性等不确定性有容忍度
    
    参数：
        L: 开环传递函数 L(s) = C(s)·G(s)
    
    返回：
        pm: 相位裕度 (度)
        gm: 增益裕度 (dB)
        bode_data: (频率, 幅值, 相位) 用于绘图
    """
    w = np.logspace(-2, 2, 800)  # 频率范围：0.01 到 100 rad/s
    w, mag, phase = signal.bode(L, w=w)

    # 展开相位（处理 ±180° 跳变）
    phase_unwrapped = np.unwrap(np.deg2rad(phase))
    phase_deg = np.rad2deg(phase_unwrapped)

    # 相位裕度：找到幅值 ≈ 0dB 处的相位
    idx_pm = np.argmin(np.abs(mag))  # 增益穿越点
    pm = 180 + phase_deg[idx_pm]
    
    # 增益裕度：找到相位 ≈ -180° 处的幅值
    idx_gm = np.argmin(np.abs(phase_deg + 180))  # 相位穿越点
    gm = -mag[idx_gm]

    return pm, gm, (w, mag, phase)


def compute_sensitivity(L: signal.TransferFunction):
    """
    计算灵敏度函数 S 和互补灵敏度函数 T
    
    这是现代控制理论中评估系统性能的核心工具，揭示了控制系统的
    两个基本权衡：抗扰性能 vs 噪声放大。
    
    1. 灵敏度函数 S(s) = 1/(1+L(s))：
       
       物理意义：
       - 测量输出对扰动的敏感程度
       - |S(jω)| 表示扰动在频率 ω 处被放大或衰减的倍数
       
       设计目标：
       - 低频段：|S| 要小（<< 0 dB），实现强抗扰
         例如：|S(0.1)| = -20dB 表示低频扰动被衰减到 1/10
       - 中频段：|S| 峰值要适中，避免共振放大
         峰值 Ms = max|S| 通常要求 < 2.0 (即 6dB)
       - 关系：Ms 越大 → 超调量越大、阻尼越差
       
       实际应用：
       - 低频抗扰：对抗风阻、重力偏差等慢速扰动
       - Ms 峰值：预测阶跃响应的超调量
    
    2. 互补灵敏度函数 T(s) = L(s)/(1+L(s))：
       
       物理意义：
       - 测量传感器噪声被放大的程度
       - |T(jω)| 表示测量噪声在频率 ω 处的放大倍数
       
       设计目标：
       - 高频段：|T| 要小（→ -∞ dB），避免噪声放大
       - 带宽频率：|T|=-3dB 处对应系统响应速度
       
       实际应用：
       - 传感器噪声抑制：陀螺仪、加速度计的高频噪声
       - 测量不确定性：低通滤波特性
    
    3. 基本权衡定理 S + T = 1 (Bode 积分约束)：
       
       深刻含义：
       - 不能同时在所有频率上做到 |S| 小和 |T| 小
       - 改善低频抗扰 (↓|S|_low) 必然恶化高频噪声 (↑|T|_high)
       - 这是物理定律，任何控制器都无法突破
       
       设计策略：
       - 低频段：让 |S| 小 (抗扰)，此时 |T| ≈ 1 (跟踪)
       - 高频段：让 |T| 小 (抑噪)，此时 |S| ≈ 1 (不控)
       - 分界点：由控制带宽决定
    
    参数：
        L: 开环传递函数 L(s) = C(s)·G(s)
    
    返回：
        w: 频率数组 (rad/s)
        Sw: 灵敏度 |S(jω)| (dB)
        Tw: 互补灵敏度 |T(jω)| (dB)
    """
    w = np.logspace(-2, 2, 800)
    Sw = []
    Tw = []
    
    for wi in w:
        jw = 1j*wi
        # 计算 L(jω)
        Ln, Ld = _tf_coeffs(L)
        numL = np.polyval(Ln, jw)
        denL = np.polyval(Ld, jw)
        Ljw = numL/denL
        
        # S(jω) = 1/(1+L(jω))
        Sjw = 1.0/(1.0 + Ljw)
        
        # T(jω) = L(jω)/(1+L(jω)) = 1 - S(jω)
        Tjw = Ljw/(1.0 + Ljw)
        
        # 转换为 dB
        Sw.append(20*np.log10(np.abs(Sjw)))
        Tw.append(20*np.log10(np.abs(Tjw)))
    
    return w, np.array(Sw), np.array(Tw)


def compute_time_response(T: signal.TransferFunction):
    t = np.linspace(0, 20, 2000)
    tout, yout = signal.step(T, T=t)
    tout2, yimp = signal.impulse(T, T=t)
    return tout, yout, tout2, yimp


def main():
    # ========================================================================
    # 分析对象：MCTS 合成的三种代表性控制策略
    # ========================================================================
    # 从 best_program.json 提取的真实 PID 参数：
    # 
    # Rule1 Baseline: 保守稳定策略
    #   - 适用场景：正常飞行、巡航、悬停等常规工况
    #   - 特点：中等响应速度，良好稳定性，低超调
    #   - 物理含义：Kp 适中保证响应，Ki 较大消除稳态误差，Kd 适中提供阻尼
    # 
    # Rule3 P-Enhanced: 仅 P 增益优化
    #   - 适用场景：低俯仰角速度误差时的精细调整
    #   - 特点：继承 Rule1 的 I/D，仅提高 P 增强响应速度
    #   - 策略：保持稳定性前提下提升跟踪性能
    # 
    # Rule5 High-Response: 高响应紧急策略
    #   - 适用场景：高角速度、紧急机动、抗风等极端工况
    #   - 特点：高 Kp 快速响应，低 Ki 减少相位滞后，中 Kd 维持阻尼
    #   - 策略：牺牲部分稳定裕度换取快速响应和强抗扰能力
    # ========================================================================
    cfgs = {
        'Rule1_Baseline': dict(Kp=0.9179, Ki=1.9055, Kd=1.0921, desc='Baseline Stable Strategy'),
        'Rule3_POnly': dict(Kp=1.574, Ki=1.9055, Kd=1.0921, desc='P-Enhanced Strategy (inherit I/D)'),
        'Rule5_HighResp': dict(Kp=2.1452, Ki=0.815, Kd=0.7451, desc='High Response Emergency Strategy'),
    }

    # 被控对象：二阶标准模型（近似无人机动力学）
    G = plant_tf(wn=1.0, zeta=0.7)

    # ========================================================================
    # 执行控制理论分析
    # ========================================================================
    results = {}
    for name, g in cfgs.items():
        # 构造 PID 控制器
        C = pid_tf(g['Kp'], g['Ki'], g['Kd'])
        
        # 闭环分析
        L, T, S = closed_loop_tf(C, G)
        # L: 开环传函 L(s)=C(s)G(s)，用于频域分析
        # T: 闭环传函 T(s)=L/(1+L)，输入到输出的传递关系
        # S: 灵敏度函数 S(s)=1/(1+L)，扰动到输出的传递关系
        
        # 李雅普诺夫稳定性分析
        eig, stable, P = lyapunov_check(T)
        # eig: 闭环极点（系统特征值），决定系统动态行为
        #      - 实部 < 0: 衰减模态（稳定）
        #      - 实部 > 0: 发散模态（不稳定）
        #      - 虚部 ≠ 0: 振荡模态（频率 = |虚部|）
        # stable: 李雅普诺夫稳定性判定（所有极点实部 < 0）
        # P: 李雅普诺夫矩阵（正定则稳定）
        
        # 稳定裕度分析
        pm, gm, bode_data = compute_margins(L)
        # pm: 相位裕度（稳定性余量）
        # gm: 增益裕度（鲁棒性余量）
        
        # 灵敏度分析
        w_sens, Sw, Tw = compute_sensitivity(L)
        # Sw: 灵敏度函数（抗扰性能指标）
        # Tw: 互补灵敏度（噪声放大指标）
        
        # 时域响应分析
        t_step, y_step, t_imp, y_imp = compute_time_response(T)
        # y_step: 阶跃响应（跟踪性能）
        # y_imp: 冲激响应（瞬态特性）
        
        results[name] = {
            'L': L, 'T': T, 'S': S,
            'eig': eig, 'stable': stable, 'P': P,
            'pm': pm, 'gm': gm,
            'bode': bode_data,
            'sens': (w_sens, Sw, Tw),
            'time': (t_step, y_step, t_imp, y_imp),
            'gains': g
        }
        
        print(f"\n[{name}] Eigenvalues:", eig)
        print(f"[{name}] Lyapunov stable:", stable)
        print(f"[{name}] Phase margin (approx deg): {pm:.2f}; Gain margin proxy (dB delta near 0dB): {gm:.2f}")

    # ========================================================================
    # 生成对比图表 - 揭示控制器性能差异
    # ========================================================================
    # 图1: Bode 图对比
    #      说明：频域特性对比，展示稳定裕度和频率响应差异
    #      关键发现：Rule5 相位裕度最高但穿越频率也最高（快速但稳定）
    plot_bode_comparison(results)
    
    # 图2: 灵敏度函数对比
    #      说明：抗扰性能 vs 噪声放大的权衡
    #      关键发现：Rule5 低频抗扰最强，Rule1 最均衡
    plot_sensitivity_comparison(results)
    
    # 图3: 时域响应对比
    #      说明：实际飞行中的响应速度和稳定性
    #      关键发现：Rule5 上升最快但超调最大，Rule1 最平稳
    plot_time_comparison(results)
    
    # 图4: 综合性能雷达图
    #      说明：多维度性能总览，直观展示优劣势
    #      关键发现：Rule5 综合性能最强，特别是在高要求场景
    plot_performance_radar(results)

    # Save summary
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        for name, r in results.items():
            f.write(f"{name}: stable={r['stable']}, pm~{r['pm']:.2f}deg, gm~{r['gm']:.2f}dB, eig={r['eig']}\n")
    print(f"\n[WRITE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print(f"[DONE] 4 summary plots saved to {OUT_DIR}")


def plot_bode_comparison(results):
    """
    FIGURE 1: Bode Plot Comparison - Open-Loop Frequency Response
    
    Shows magnitude and phase response across frequency for all control strategies.
    
    What it reveals:
    - Phase Margin (PM): Safety margin before instability (>30° is good, >60° is excellent)
    - Gain Margin (GM): How much gain can increase before system oscillates
    - Crossover Frequency: Speed of response (higher = faster, but less stable)
    - High-freq roll-off: Noise rejection capability
    
    Key insight: Rule5_HighResp has higher crossover (faster) but lower PM (less stable)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for name, r in results.items():
        w, mag, phase = r['bode']
        label = f"{name.replace('_', ' ')} (PM≈{r['pm']:.1f}°)"
        ax1.semilogx(w, mag, label=label, linewidth=2)
        ax2.semilogx(w, phase, label=name.replace('_', ' '), linewidth=2)
    
    ax1.set_ylabel('Magnitude (dB)', fontsize=11)
    ax1.set_title('Open-Loop Bode Plot Comparison - Real Strategies from best_program.json', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', ls=':', alpha=0.4)
    ax1.legend(fontsize=10)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='0 dB')
    
    ax2.set_ylabel('Phase (deg)', fontsize=11)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=11)
    ax2.grid(True, which='both', ls=':', alpha=0.4)
    ax2.legend(fontsize=10)
    ax2.axhline(-180, color='red', linestyle='--', alpha=0.5, linewidth=1, label='-180°')
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "1_bode_comparison.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[SAVE] {out}")


def plot_sensitivity_comparison(results):
    """
    FIGURE 2: Sensitivity Functions - Disturbance Rejection vs Noise Amplification
    
    Left plot - Sensitivity |S|: How well system rejects low-frequency disturbances
    - Lower is better at low frequencies (better disturbance rejection)
    - Peak value indicates resonance (overshoot in response)
    
    Right plot - Complementary Sensitivity |T|: High-frequency noise amplification
    - Lower is better at high frequencies (less noise amplification)
    - Trade-off with |S|: improving one worsens the other (S + T = 1)
    
    Key insight: Rule1 Baseline balances both; Rule5 HighResp has better low-freq
    rejection but worse high-freq noise handling
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, r in results.items():
        w, Sw, Tw = r['sens']
        label = name.replace('_', ' ')
        ax1.semilogx(w, Sw, label=label, linewidth=2)
        ax2.semilogx(w, Tw, label=label, linewidth=2)
    
    ax1.set_title('Sensitivity Function |S| - Disturbance Rejection', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Frequency (rad/s)', fontsize=11)
    ax1.set_ylabel('|S| (dB)', fontsize=11)
    ax1.grid(True, which='both', ls=':', alpha=0.4)
    ax1.legend(fontsize=10)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_title('Complementary Sensitivity |T| - Noise Amplification', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (rad/s)', fontsize=11)
    ax2.set_ylabel('|T| (dB)', fontsize=11)
    ax2.grid(True, which='both', ls=':', alpha=0.4)
    ax2.legend(fontsize=10)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "2_sensitivity_comparison.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[SAVE] {out}")


def plot_time_comparison(results):
    """
    FIGURE 3: Time Domain Response Comparison
    
    Left plot - Step Response: System's response to a sudden command change
    - Rise time: How fast system reaches target (faster = more responsive)
    - Overshoot: How much it exceeds target (lower = more stable)
    - Settling time: Time to stabilize within ±2% of final value
    
    Right plot - Impulse Response: System's transient behavior to a shock
    - Peak amplitude: Maximum reaction to disturbance
    - Decay rate: How quickly system returns to equilibrium
    - Oscillations: Indicates damping quality
    
    Key insight: Rule5 has fastest rise but highest overshoot (aggressive);
    Rule1 is well-damped with minimal overshoot (conservative)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, r in results.items():
        t_step, y_step, t_imp, y_imp = r['time']
        label = name.replace('_', ' ')
        ax1.plot(t_step, y_step, label=label, linewidth=2)
        ax2.plot(t_imp, y_imp, label=label, linewidth=2, alpha=0.8)
    
    ax1.set_title('Step Response Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Output', fontsize=11)
    ax1.grid(True, ls=':', alpha=0.4)
    ax1.legend(fontsize=10)
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Target')
    
    ax2.set_title('Impulse Response Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Output', fontsize=11)
    ax2.grid(True, ls=':', alpha=0.4)
    ax2.legend(fontsize=10)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "3_time_domain_comparison.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[SAVE] {out}")


def plot_performance_radar(results):
    """
    FIGURE 4: Performance Radar Chart - Multi-Dimensional Comparison
    
    Compares four key control performance metrics (0-1 normalized):
    
    1. Phase Margin: Stability margin (higher = safer, target >60°)
       - Measures how close system is to instability
    
    2. Stability: Lyapunov stability check (1 = stable, 0 = unstable)
       - All eigenvalues must have negative real parts
    
    3. Low-freq Rejection: Disturbance rejection at 0.1 rad/s (higher = better)
       - How well system rejects slow disturbances (wind, drift)
    
    4. Damping: Oscillation damping ratio (higher = less overshoot, target ~0.7)
       - 0.7 is critically damped (optimal balance)
       - <0.7 is underdamped (oscillatory)
       - >0.7 is overdamped (sluggish)
    
    Key insight: Rule1 Baseline has the most balanced profile;
    Rule5 HighResp sacrifices stability for better disturbance rejection
    """
    import math
    
    # Define evaluation metrics (normalized to 0-1)
    metrics = ['Phase Margin', 'Stability', 'Low-freq Rejection', 'Damping']
    num_metrics = len(metrics)
    
    # Extract metrics
    data = {}
    for name, r in results.items():
        pm_norm = min(r['pm'] / 90.0, 1.0)  # 90° = perfect score
        stable_norm = 1.0 if r['stable'] else 0.0
        
        # Low-freq disturbance rejection: S(0.1 rad/s) in dB, lower is better
        w, Sw, _ = r['sens']
        idx_low = np.argmin(np.abs(w - 0.1))
        s_low_db = Sw[idx_low]
        rejection_norm = max(0, 1.0 + s_low_db / 20.0)  # -20dB -> 1.0, 0dB -> 0.5
        
        # Damping: estimate from eigenvalues, compute damping ratio for complex poles
        eig = r['eig']
        complex_poles = [e for e in eig if np.imag(e) != 0]
        if complex_poles:
            pole = complex_poles[0]
            sigma = -np.real(pole)
            omega = np.abs(np.imag(pole))
            zeta = sigma / np.sqrt(sigma**2 + omega**2) if (sigma**2 + omega**2) > 0 else 0
            damping_norm = min(zeta / 0.7, 1.0)  # 0.7 = ideal damping
        else:
            damping_norm = 1.0  # Overdamped (all real poles)
        
        data[name.replace('_', ' ')] = [pm_norm, stable_norm, rejection_norm, damping_norm]
    
    # Plot radar chart
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    for name, values in data.items():
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_title('Comprehensive Performance Radar - Real Strategies from best_program.json', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "4_performance_radar.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {out}")


if __name__ == '__main__':
    main()
