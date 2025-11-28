&emsp;&emsp;无人机 (UAV) 在复杂动态环境下的自主控制，对控制器的自适应能力提出了严苛要求。
    
&emsp;&emsp;传统的PID控制器因其固定的增益参数，难以在多变的高速机动、阵风干扰等复杂飞行任务中维持最优性能。

&emsp;&emsp;近年来深度强化学习 (DRL) 作为一种强大的自适应方法取得了显著成功，但其对“黑盒”神经网络的依赖，带来了阻碍其在现实世界，特别是安全关键领域的应用的三个核心挑战：缺乏可解释性，导致策略难以被信任和验证。资源消耗巨大，与无人机机载的低功耗微控制器不兼容。泛化能力有限，难以将在一个场景训练的策略直接应用于新场景。

&emsp;&emsp;为了解决这一困境，我们提出了 π-Flight，一个旨在自动生成可解释符号程序作为控制器策略的框架。π-Flight的核心思想是，将复杂的控制逻辑表达为一系列人类可读的程序与规则，从而兼顾高性能与高可信度。

## SCG 原生环境测试

按照项目约定，所有控制器（PID/LQR/π-Flight 程序）都需要在 [safe-control-gym](https://github.com/utiasDSL/safe-control-gym) 的原生环境上做评估，仅训练阶段保留在 Isaac Sim。以下是快速操作流程：

1. **安装 safe-control-gym**：
	```bash
	source .venv/bin/activate
	pip install -e /path/to/safe-control-gym  # 或者从外部镜像 git clone 后本地安装
	```
	无法直接访问 GitHub 时，请先准备好离线包或镜像源，再执行 `pip install -e`。

2. **运行 PID（默认）测试**：
	```bash
	python scripts/eval_safecontrol_piflight.py --env-id quadrotor_tracking --controller pid --episodes 3
	```

3. **运行 LQR 测试**：
	```bash
	python scripts/eval_safecontrol_piflight.py --env-id quadrotor_tracking --controller lqr --episodes 3
	```

4. **运行 π-Flight 程序测试**（action 仍由 SCG 环境执行）：
	```bash
	python scripts/eval_safecontrol_piflight.py \
		 --env-id quadrotor_tracking \
		 --controller piflight \
		 --program-json results/demo_program.json \
		 --device cuda:0
	```

`scripts/eval_safecontrol_piflight.py` 会自动调用 `safe_control_gym.envs.make`（若可用），否则退回到 `gymnasium.make`。默认 action space 为 `[Fz, Tx, Ty, Tz]`，可通过 `--action-space motors` 输出归一化桨速。命令结束后会输出 RMSE、最大误差、能量、约束违例和成功率等指标。
