from __future__ import annotations
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover - torch is optional at runtime
    # Provide minimal shims so static analyzers don't flag attribute-missing errors
    class _NNShim:  # pragma: no cover
        class Module:  # type: ignore[misc]
            pass
        class Linear:  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, x):
                return x
    class _FShim:  # pragma: no cover
        @staticmethod
        def relu(x):
            return x
    class _TorchShim:  # pragma: no cover
        @staticmethod
        def sigmoid(x):
            return x
        @staticmethod
        def tensor(*args, **kwargs):
            raise RuntimeError('PyTorch not available')
        @staticmethod
        def no_grad():
            class _Ctx:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False
            return _Ctx()
        @staticmethod
        def load(*args, **kwargs):
            raise RuntimeError('PyTorch not available')
        @staticmethod
        def save(*args, **kwargs):
            raise RuntimeError('PyTorch not available')
    torch = _TorchShim()  # type: ignore
    nn = _NNShim()        # type: ignore
    F = _FShim()          # type: ignore
    _TORCH_OK = False


if _TORCH_OK:
    class TRNNModel(nn.Module):  # type: ignore[misc]
        """
        A tiny MLP that maps per-dimension features -> (radius_frac, sigma_frac).
        Features per-dim (default minimal): [norm_x, width, 1.0].
        Outputs are in (0,1) via sigmoid.
        """
        def __init__(self, in_dim: int = 3, hidden: int = 32):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.out(x))  # [B,2] in (0,1)
            return x
else:
    class TRNNModel:  # type: ignore[misc]
        pass


def load_trnn_model(path: str, device: str = 'cpu') -> TRNNModel:
    assert _TORCH_OK and (torch is not None), 'PyTorch not available'
    model = TRNNModel()  # type: ignore[call-arg]
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        model.load_state_dict(sd['state_dict'])  # type: ignore[attr-defined]
    else:
        model.load_state_dict(sd)  # type: ignore[attr-defined]
    model.to(device)  # type: ignore[attr-defined]
    model.eval()      # type: ignore[attr-defined]
    return model


def save_trnn_model(model: TRNNModel, path: str):
    assert _TORCH_OK and (torch is not None), 'PyTorch not available'
    torch.save({'state_dict': model.state_dict()}, path)  # type: ignore[attr-defined]
