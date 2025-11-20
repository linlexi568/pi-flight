"""Hard control-theoretic constraints for DSL programs.

This module encodes channel-specific variable whitelists and helpers that
can be reused across the MCTS search as well as the evaluator to make sure
unphysical control laws never get simulated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

try:
    from ..core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode
except Exception:  # pragma: no cover - fallback when executed as script
    from core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore

# Outputs that appear on the left-hand side of set operations
CONTROL_OUTPUTS = {"u_fz", "u_tx", "u_ty", "u_tz"}

# Penalty used when a program violates the hard constraints.
HARD_CONSTRAINT_PENALTY = -1e6

# Base feature groups to keep definitions compact
POSITION_ERRS = {
    "pos_err_x",
    "pos_err_y",
    "pos_err_z",
    "pos_err_xy",
    "pos_err_z_abs",
}
VELOCITIES = {
    "vel_x",
    "vel_y",
    "vel_z",
    "vel_err",
}
ANGULAR_VELS = {
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "ang_vel",
    "ang_vel_mag",
}
ATTITUDE_ERRS = {
    "err_p_roll",
    "err_p_pitch",
    "err_p_yaw",
    "rpy_err_mag",
}
INTEGRALS = {
    "err_i_x",
    "err_i_y",
    "err_i_z",
    "err_i_roll",
    "err_i_pitch",
    "err_i_yaw",
}
DERIVATIVES = {
    "err_d_x",
    "err_d_y",
    "err_d_z",
    "err_d_roll",
    "err_d_pitch",
    "err_d_yaw",
}

# Channel-specific whitelists. Each set enumerates the state variables a given
# actuator is allowed to reference. The sets are intentionally redundant so the
# controller still has enough freedom while explicitly ruling out pathological
# couplings such as feeding yaw error into the thrust channel.
CHANNEL_ALLOWED_INPUTS: Dict[str, Set[str]] = {
    # Collective thrust: vertical position/velocity errors + roll/pitch attitude
    # for leveling. No yaw coupling allowed.
    "u_fz": {
        *POSITION_ERRS,
        *VELOCITIES,
        "err_p_roll",
        "err_p_pitch",
        "err_d_roll",
        "err_d_pitch",
        "err_i_z",
    },
    # Roll torque: roll attitude + lateral/posture errors (x axis) and their rates.
    "u_tx": {
        "err_p_roll",
        "err_d_roll",
        "err_i_x",
        "pos_err_x",
        "pos_err_y",
        "vel_x",
        "vel_y",
        "vel_err",
        "pos_err_xy",
        "ang_vel_x",
    },
    # Pitch torque: pitch attitude + longitudinal errors.
    "u_ty": {
        "err_p_pitch",
        "err_d_pitch",
        "err_i_y",
        "pos_err_x",
        "pos_err_y",
        "vel_x",
        "vel_y",
        "vel_err",
        "pos_err_xy",
        "ang_vel_y",
    },
    # Yaw torque: yaw attitude + angular velocity; allow mild coupling with xy
    # velocity for damping but forbid direct vertical terms.
    "u_tz": {
        "err_p_yaw",
        "err_d_yaw",
        "err_i_yaw",
        "vel_x",
        "vel_y",
        "vel_err",
        "pos_err_x",
        "pos_err_y",
        "pos_err_xy",
        "ang_vel_z",
    },
}


def _collect_state_variables(node: ProgramNode, bucket: Set[str]) -> None:
    """Recursively collect state variable names referenced by an AST node."""
    if node is None:
        return
    if isinstance(node, TerminalNode):
        val = node.value
        if isinstance(val, str) and val not in CONTROL_OUTPUTS:
            bucket.add(val)
        return
    if isinstance(node, UnaryOpNode):
        _collect_state_variables(node.child, bucket)
        return
    if isinstance(node, BinaryOpNode):
        _collect_state_variables(node.left, bucket)
        _collect_state_variables(node.right, bucket)
        return
    if isinstance(node, IfNode):
        _collect_state_variables(node.condition, bucket)
        _collect_state_variables(node.then_branch, bucket)
        _collect_state_variables(node.else_branch, bucket)


def allowed_variables_for_channel(channel: str, available: List[str]) -> List[str]:
    """Return the subset of DSL variables that a channel is allowed to use."""
    allowed = CHANNEL_ALLOWED_INPUTS.get(channel)
    if not allowed:
        return list(available)
    subset = [v for v in available if v in allowed]
    return subset


def validate_action_channel(action_node: ProgramNode) -> Tuple[bool, str]:
    """Validate a single Binary(set, u_*, expr) node against the whitelist."""
    if not isinstance(action_node, BinaryOpNode) or action_node.op != 'set':
        return True, ""
    left = action_node.left
    if not isinstance(left, TerminalNode) or not isinstance(left.value, str):
        return False, "action missing output terminal"
    channel = left.value
    if channel not in CONTROL_OUTPUTS:
        return False, f"unknown actuator '{channel}'"
    expr = action_node.right
    used_vars: Set[str] = set()
    _collect_state_variables(expr, used_vars)
    allowed = CHANNEL_ALLOWED_INPUTS.get(channel)
    if allowed is None:
        return True, ""
    illegal = sorted(v for v in used_vars if v not in allowed)
    if illegal:
        return False, f"{channel} references disallowed inputs: {illegal}"
    return True, ""


def validate_program(program: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Check whether every action in the program satisfies hard constraints."""
    for rule_idx, rule in enumerate(program or []):
        actions = rule.get('action', []) if isinstance(rule, dict) else []
        for action_idx, action in enumerate(actions):
            ok, reason = validate_action_channel(action)
            if not ok:
                return False, f"rule#{rule_idx}/action#{action_idx}: {reason}"
    return True, ""
