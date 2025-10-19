"""Attention-based Gain Scheduling Network."""

from .attn_model import AttnGainNet
from .attn_controller import AttnController

__all__ = ['AttnGainNet', 'AttnController']
