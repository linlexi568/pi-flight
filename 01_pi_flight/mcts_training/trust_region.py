from __future__ import annotations
from typing import List, Tuple, Optional

try:
    # lazy import to avoid hard dependency
    from ..nn_training.trust_region.tr_nn import load_trnn_model  # type: ignore
except Exception:
    try:
        from ..nn_training.trust_region.tr_nn import load_trnn_model  # type: ignore
    except Exception:
        load_trnn_model = None  # type: ignore


class TrustRegionManager:
    """
    Minimal trust-region manager to produce per-dimension narrow windows for CMA or local tuning.

    Strategy (heuristic, NN-ready):
    - Center at current parameter value (mu = base).
    - Radius r_i = clamp(frac * (hi-lo), min_frac*(hi-lo), max_frac*(hi-lo)).
    - Sigma per dim = max(1e-8, r_i * sigma_frac).
    - Ensure x0 strictly inside [low_i, high_i] with a tiny epsilon margin.

    This class is NN-ready: you can later replace the heuristic in suggest_for_vector
    with a model-driven radius and sigma predictor while keeping the same interface.
    """

    def __init__(self, args=None):
        # fractions relative to global bounds
        self.enabled: bool = True
        self.radius_frac: float = float(getattr(args, 'tr_radius_frac', 0.25) if args is not None else 0.25)
        self.min_radius_frac: float = float(getattr(args, 'tr_min_radius_frac', 0.05) if args is not None else 0.05)
        self.max_radius_frac: float = float(getattr(args, 'tr_max_radius_frac', 0.60) if args is not None else 0.60)
        self.sigma_frac: float = float(getattr(args, 'tr_sigma_frac', 0.33) if args is not None else 0.33)
        # NN options
        self.nn_used: bool = False
        self._nn = None
        if args is not None and bool(getattr(args, 'tr_nn_enable', False)) and load_trnn_model is not None:
            try:
                path = str(getattr(args, 'tr_nn_path', '01_pi_flight/results/trnn.pt'))
                device = str(getattr(args, 'tr_nn_device', 'cpu'))
                self._nn = load_trnn_model(path, device=device)
                self.nn_used = True
            except Exception:
                self._nn = None
                self.nn_used = False

    def suggest_for_vector(
        self,
        base_vec: List[float],
        global_low: List[float],
        global_high: List[float],
        mode: str = 'joint'
    ) -> Tuple[List[float], List[float], List[float], List[float], bool, float, bool]:
        """
    Return (x0, low, high, sigma_list, used_flag, radius_mean, nn_used).
        - x0: initial point (clamped into [low, high]).
        - low/high: per-dimension trust bounds, subset of global bounds.
        - sigma_list: per-dimension initial stddevs.
        - used_flag: True if TR is active.
        - radius_mean: mean absolute radius across dimensions (for logging).
        """
        n = len(base_vec)
        if n == 0:
            return list(base_vec), list(global_low), list(global_high), [1e-3]*0, False, float('nan'), False
        eps = 1e-8
        x0 = list(map(float, base_vec))
        lo_out: List[float] = [0.0]*n
        hi_out: List[float] = [0.0]*n
        sigmas: List[float] = [0.0]*n
        r_sum = 0.0
        # optional: prepare NN features per-dim
        nn_radius_frac = None
        nn_sigma_frac = None
        if self._nn is not None:
            try:
                import torch  # type: ignore
                feats = []
                for i in range(n):
                    glo = float(global_low[i]); ghi = float(global_high[i])
                    if glo > ghi:
                        glo, ghi = ghi, glo
                    width = max(1e-12, (ghi - glo))
                    x = float(base_vec[i])
                    # normalized to [0,1] within global bounds
                    norm_x = (x - glo) / width
                    feats.append([norm_x, width, 1.0])
                X = torch.tensor(feats, dtype=torch.float32)
                with torch.no_grad():
                    out = self._nn(X)  # type: ignore[operator]
                # map to fractions
                nn_radius_frac = out[:, 0].clamp(0.0, 1.0).cpu().numpy().tolist()
                nn_sigma_frac = out[:, 1].clamp(0.0, 1.0).cpu().numpy().tolist()
            except Exception:
                nn_radius_frac = None
                nn_sigma_frac = None
        for i in range(n):
            glo = float(global_low[i]); ghi = float(global_high[i])
            if glo > ghi:
                glo, ghi = ghi, glo
            width = max(1e-12, (ghi - glo))
            # radius absolute size
            r_frac = self.radius_frac
            if nn_radius_frac is not None:
                # clamp NN outputs into [min,max]
                r_frac = max(self.min_radius_frac, min(self.max_radius_frac, float(nn_radius_frac[i])))
            r = r_frac * width
            r_min = self.min_radius_frac * width
            r_max = self.max_radius_frac * width
            if r < r_min:
                r = r_min
            if r > r_max:
                r = r_max
            mu = float(base_vec[i])
            lo_i = max(glo, mu - r)
            hi_i = min(ghi, mu + r)
            # If collapsed by global bounds, push a minimal thickness
            if (hi_i - lo_i) < (1e-12):
                mid = (glo + ghi) * 0.5
                lo_i = max(glo, mid - r_min * 0.5)
                hi_i = min(ghi, mid + r_min * 0.5)
                if (hi_i - lo_i) < 1e-12:
                    # final fallback
                    lo_i = max(glo, ghi - 1e-12)
                    hi_i = ghi
            x = mu
            if x <= lo_i:
                x = lo_i + eps * (hi_i - lo_i)
            elif x >= hi_i:
                x = hi_i - eps * (hi_i - lo_i)
            s_frac = self.sigma_frac
            if nn_sigma_frac is not None:
                s_frac = max(1e-4, min(1.0, float(nn_sigma_frac[i])))
            sigma_i = max(1e-8, s_frac * (hi_i - lo_i))
            x0[i] = x
            lo_out[i] = lo_i
            hi_out[i] = hi_i
            sigmas[i] = sigma_i
            r_sum += (hi_i - lo_i) * 0.5
        r_mean = r_sum / max(1, n)
        return x0, lo_out, hi_out, sigmas, True, float(r_mean), bool(self.nn_used and (nn_radius_frac is not None))
