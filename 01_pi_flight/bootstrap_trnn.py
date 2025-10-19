from __future__ import annotations
"""Shim to nn_training/trust_region/bootstrap_trnn.py"""
from .nn_training.trust_region.bootstrap_trnn import *  # type: ignore

if __name__ == '__main__':
    from .nn_training.trust_region.bootstrap_trnn import main  # type: ignore
    main()
