from __future__ import annotations
import os
import math
from typing import Tuple

from .tr_nn import TRNNModel, save_trnn_model  # type: ignore


def logit(p: float) -> float:
    p = max(1e-6, min(1-1e-6, p))
    return math.log(p/(1.0-p))


def build_constant_model(radius_frac: float = 0.25, sigma_frac: float = 0.33) -> TRNNModel:
    """
    Create a TRNNModel that outputs constant (radius_frac, sigma_frac) for any input
    by zeroing all weights and setting the final bias to logit(target).
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError('PyTorch required to build constant model') from e
    m = TRNNModel()  # type: ignore[call-arg]
    with torch.no_grad():
        for p in m.parameters():  # type: ignore[attr-defined]
            p.zero_()
        # Set final layer bias so sigmoid(out) gives desired constants
        b = m.out.bias  # type: ignore[attr-defined]
        b[...] = torch.tensor([logit(radius_frac), logit(sigma_frac)], dtype=b.dtype)
    return m


def main(path: str = '01_pi_flight/results/trnn.pt', radius_frac: float = 0.25, sigma_frac: float = 0.33):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    model = build_constant_model(radius_frac, sigma_frac)
    save_trnn_model(model, path)
    print(f"[TRNN] Bootstrapped constant model -> {path} (radius_frac={radius_frac}, sigma_frac={sigma_frac})")


if __name__ == '__main__':
    main()
