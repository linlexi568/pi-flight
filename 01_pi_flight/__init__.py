"""pi_flight package

Programmatic flight controller search via DSL + MCTS. This is the canonical
package going forward (renamed from 01_pi_light). It exposes the same API.

This module is resilient to internal folder reorganization. It will try to
import symbols from both legacy flat layout and the new split layout:
  - mcts:   .mcts or .mcts_training.mcts
  - nn:     (not exported here; training utilities live under .nn_training)
  - cma-es: (not exported here; tooling can live under .cma_training)
"""
from .dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode

# Stable import for segmented controller
from .segmented_controller import PiLightSegmentedPIDController

# Try new split path first, then legacy fallback
try:
    from .mcts_training.mcts import MCTS_Agent  # type: ignore
except Exception:
    from .mcts import MCTS_Agent  # type: ignore

__all__ = [
    'ProgramNode','TerminalNode','UnaryOpNode','BinaryOpNode','IfNode',
    'MCTS_Agent','PiLightSegmentedPIDController'
]
