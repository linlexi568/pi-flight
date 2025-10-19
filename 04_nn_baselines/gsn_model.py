import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GainSchedulingNet(nn.Module):
    """简单的增益调度网络：输入状态向量，输出 P/I/D 乘数 (已映射至安全区间)。"""
    def __init__(self, state_dim: int, hidden_dims=(128, 128),
                 p_range: Tuple[float, float] = (0.6, 2.5),
                 i_range: Tuple[float, float] = (0.5, 2.0),
                 d_range: Tuple[float, float] = (0.5, 2.0)):
        super().__init__()
        self.p_range = p_range
        self.i_range = i_range
        self.d_range = d_range
        dims = [state_dim] + list(hidden_dims)
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 3)  # raw outputs for P,I,D

    def forward(self, state: torch.Tensor):
        x = self.backbone(state)
        raw = self.head(x)
        # sigmoid -> (0,1)
        sig = torch.sigmoid(raw)
        p = self.p_range[0] + (self.p_range[1] - self.p_range[0]) * sig[..., 0]
        i = self.i_range[0] + (self.i_range[1] - self.i_range[0]) * sig[..., 1]
        d = self.d_range[0] + (self.d_range[1] - self.d_range[0]) * sig[..., 2]
        return torch.stack([p, i, d], dim=-1)

if __name__ == '__main__':
    net = GainSchedulingNet(state_dim=18)
    dummy = torch.randn(4, 18)
    out = net(dummy)
    print(out.shape, out.min().item(), out.max().item())
