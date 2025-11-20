"""GPU 表达式执行器，实现 DSL 程序的全 GPU 求值。"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

try:  # pragma: no cover
    from core.dsl import (  # type: ignore
        TerminalNode,
        UnaryOpNode,
        BinaryOpNode,
        IfNode,
        SAFE_VALUE_MIN,
        SAFE_VALUE_MAX,
        MIN_EMA_ALPHA,
        MAX_EMA_ALPHA,
        MAX_DELAY_STEPS,
        MAX_DIFF_STEPS,
        MAX_RATE_LIMIT,
    )
except Exception:  # pragma: no cover
    from ..core.dsl import (  # type: ignore
        TerminalNode,
        UnaryOpNode,
        BinaryOpNode,
        IfNode,
        SAFE_VALUE_MIN,
        SAFE_VALUE_MAX,
        MIN_EMA_ALPHA,
        MAX_EMA_ALPHA,
        MAX_DELAY_STEPS,
        MAX_DIFF_STEPS,
        MAX_RATE_LIMIT,
    )


class GPUProgramExecutor:
    """GPU 版 DSL 执行器，支持表达式和常量程序。"""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        self.program_cache: Dict[str, Tensor] = {}
        self._batches: Dict[int, Dict[str, Any]] = {}
        self._next_batch_token = 1
        self._zero = torch.tensor(0.0, device=self.device)
        self._one = torch.tensor(1.0, device=self.device)

    # ------------------------------------------------------------------
    # 常量程序预编译 (兼容旧版)
    # ------------------------------------------------------------------
    def compile_constant_programs(self, programs: List[List[Dict[str, Any]]]) -> Optional[Tensor]:
        constants: List[List[float]] = []
        for prog in programs:
            forces = self._extract_constant_forces(prog)
            if forces is None:
                return None
            constants.append(forces)
        return torch.tensor(constants, device=self.device, dtype=torch.float32)

    def _extract_constant_forces(self, program: List[Dict[str, Any]]) -> Optional[List[float]]:
        mapping = {"u_fz": 0, "u_tx": 1, "u_ty": 2, "u_tz": 3}
        forces = [0.0, 0.0, 0.0, 0.0]
        for rule in program or []:
            if rule.get("op") != "set":
                return None
            var = rule.get("var")
            if var not in mapping:
                return None
            expr = rule.get("expr", {})
            if expr.get("type") != "const":
                return None
            forces[mapping[var]] = float(expr.get("value", 0.0))
        return forces

    def apply_constant_forces_gpu(self, compiled: Tensor, batch_size: int, num_envs: int) -> Tensor:
        actions = torch.zeros((num_envs, 6), device=self.device, dtype=torch.float32)
        actions[:batch_size, 2:6] = compiled[:batch_size]
        return actions

    # ------------------------------------------------------------------
    # DSL 执行主流程
    # ------------------------------------------------------------------
    def prepare_batch(self, programs: List[List[Dict[str, Any]]]) -> int:
        token = self._next_batch_token
        self._next_batch_token += 1
        self._batches[token] = {
            "programs": programs,
            "node_states": [{} for _ in programs],
        }
        return token

    def release_batch(self, token: int) -> None:
        self._batches.pop(token, None)

    def reset_state(self) -> None:
        self._batches.clear()
        self._next_batch_token = 1

    def evaluate(
        self,
        token: int,
        state_tensors: Dict[str, Tensor],
        use_u_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        batch = self._batches.get(token)
        if batch is None:
            raise RuntimeError("GPUProgramExecutor batch token 无效")

        programs: List[List[Dict[str, Any]]] = batch["programs"]
        node_states: List[Dict[int, Any]] = batch["node_states"]
        batch_size = len(programs)
        outputs = torch.zeros((batch_size, 4), device=self.device)

        use_u_mask = use_u_mask.to(self.device).bool()
        if active_mask is None:
            active_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        else:
            active_mask = active_mask.to(self.device).bool()

        for idx, program in enumerate(programs):
            if not use_u_mask[idx] or not active_mask[idx]:
                continue
            node_cache = node_states[idx]
            fz = self._zero
            tx = self._zero
            ty = self._zero
            tz = self._zero

            for rule in program or []:
                cond_mask = self._one
                cond = rule.get("condition")
                if cond is not None:
                    cond_val = self._eval_node(cond, state_tensors, idx, node_cache)
                    cond_mask = torch.where(cond_val > 0.0, self._one, self._zero)

                for action in rule.get("action", []) or []:
                    if not isinstance(action, BinaryOpNode) or action.op != "set":
                        continue
                    left = getattr(action, "left", None)
                    if not isinstance(left, TerminalNode):
                        continue
                    key = str(getattr(left, "value", ""))
                    if key not in ("u_fz", "u_tx", "u_ty", "u_tz"):
                        continue
                    value = self._eval_node(getattr(action, "right", None), state_tensors, idx, node_cache)
                    value = value * cond_mask
                    if key == "u_fz":
                        fz = fz + value
                    elif key == "u_tx":
                        tx = tx + value
                    elif key == "u_ty":
                        ty = ty + value
                    elif key == "u_tz":
                        tz = tz + value

            outputs[idx, 0] = torch.clamp(fz, -5.0, 5.0)
            outputs[idx, 1] = torch.clamp(tx, -0.02, 0.02)
            outputs[idx, 2] = torch.clamp(ty, -0.02, 0.02)
            outputs[idx, 3] = torch.clamp(tz, -0.01, 0.01)

        return outputs

    # ------------------------------------------------------------------
    # 节点求值
    # ------------------------------------------------------------------
    def _eval_node(
        self,
        node: Any,
        state_tensors: Dict[str, Tensor],
        idx: int,
        cache: Dict[int, Any],
    ) -> Tensor:
        if node is None:
            return self._zero
        node_id = id(node)

        if isinstance(node, (int, float)):
            return torch.tensor(float(node), device=self.device)
        if isinstance(node, TerminalNode):
            if isinstance(node.value, str):
                tensor = state_tensors.get(node.value)
                if tensor is None:
                    return self._zero
                return tensor[idx]
            return torch.tensor(float(node.value), device=self.device)
        if isinstance(node, UnaryOpNode):
            child = self._eval_node(node.child, state_tensors, idx, cache)
            return self._eval_unary(str(getattr(node, "op", "")), child, node_id, cache)
        if isinstance(node, BinaryOpNode):
            left = self._eval_node(node.left, state_tensors, idx, cache)
            right = self._eval_node(node.right, state_tensors, idx, cache)
            return self._eval_binary(str(node.op), left, right)
        if isinstance(node, IfNode):
            cond = self._eval_node(node.condition, state_tensors, idx, cache)
            then_val = self._eval_node(node.then_branch, state_tensors, idx, cache)
            else_val = self._eval_node(node.else_branch, state_tensors, idx, cache)
            return torch.where(cond > 0.0, then_val, else_val)
        if hasattr(node, "evaluate"):
            try:
                value = node.evaluate({})  # type: ignore[arg-type]
                return torch.tensor(float(value), device=self.device)
            except Exception:
                return self._zero
        return self._zero

    def _eval_unary(self, op: str, value: Tensor, node_id: int, cache: Dict[int, Any]) -> Tensor:
        if op == "abs":
            return torch.abs(value)
        if op == "sign":
            return torch.sign(value)
        if op == "sin":
            return torch.sin(value)
        if op == "cos":
            return torch.cos(value)
        if op == "tan":
            return torch.clamp(torch.tan(value), -10.0, 10.0)
        if op == "log1p":
            return torch.log1p(torch.abs(value))
        if op == "sqrt":
            return torch.sqrt(torch.abs(value))

        prefix, *args = op.split(":")
        if prefix == "ema":
            alpha = float(args[0]) if args else 0.2
            alpha = min(max(alpha, MIN_EMA_ALPHA), MAX_EMA_ALPHA)
            prev = cache.get(node_id)
            if prev is None:
                prev = self._zero
            result = (1.0 - alpha) * prev + alpha * value
            cache[node_id] = result
            return result
        if prefix == "delay":
            steps = int(float(args[0])) if args else 1
            steps = max(1, min(MAX_DELAY_STEPS, steps))
            buf: deque = cache.get(node_id)
            if buf is None or not isinstance(buf, deque) or buf.maxlen != steps:
                buf = deque(maxlen=steps)
            out = buf[-1] if len(buf) == steps else self._zero
            buf.append(value)
            cache[node_id] = buf
            return out
        if prefix == "diff":
            steps = int(float(args[0])) if args else 1
            steps = max(1, min(MAX_DIFF_STEPS, steps))
            buf: deque = cache.get(node_id)
            if buf is None or not isinstance(buf, deque) or buf.maxlen != steps:
                buf = deque(maxlen=steps)
            prev = buf[-1] if len(buf) == steps else value
            buf.append(value)
            cache[node_id] = buf
            return value - prev
        if prefix in ("rate", "rate_limit"):
            rate = float(args[0]) if args else 1.0
            rate = min(max(0.01, rate), MAX_RATE_LIMIT)
            prev = cache.get(node_id)
            if prev is None:
                prev = self._zero
            lo = prev - rate
            hi = prev + rate
            result = torch.min(torch.max(value, lo), hi)
            cache[node_id] = result
            return result
        if prefix == "clamp":
            lo = float(args[0]) if len(args) >= 1 else SAFE_VALUE_MIN
            hi = float(args[1]) if len(args) >= 2 else SAFE_VALUE_MAX
            if lo > hi:
                lo, hi = hi, lo
            return torch.clamp(value, lo, hi)
        if prefix == "deadzone":
            eps = float(args[0]) if args else 0.01
            eps = min(max(0.0, eps), 1.0)
            mask = torch.abs(value) <= eps
            adjusted = value - torch.sign(value) * eps
            return torch.where(mask, self._zero, adjusted)
        if prefix in ("smooth", "smoothstep"):
            scale = float(args[0]) if args else 1.0
            scale = min(max(1e-3, scale), 2.0)
            return scale * torch.tanh(value / scale)
        return value

    def _eval_binary(self, op: str, left: Tensor, right: Tensor) -> Tensor:
        eps = 1e-6
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            denom = torch.where(torch.abs(right) > eps, right, torch.where(right >= 0, torch.tensor(eps, device=self.device), torch.tensor(-eps, device=self.device)))
            return left / denom
        if op == "max":
            return torch.maximum(left, right)
        if op == "min":
            return torch.minimum(left, right)
        if op == ">":
            return torch.where(left > right, self._one, self._zero)
        if op == "<":
            return torch.where(left < right, self._one, self._zero)
        if op == "==":
            return torch.where(torch.abs(left - right) < eps, self._one, self._zero)
        if op == "!=":
            return torch.where(torch.abs(left - right) >= eps, self._one, self._zero)
        return self._zero

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def quat_to_rpy_gpu(self, quat: Tensor) -> Tensor:
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=1)


__all__ = ["GPUProgramExecutor"]
