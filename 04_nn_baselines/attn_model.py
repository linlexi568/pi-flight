import math
import torch
import torch.nn as nn
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding for Transformer inputs.
    Expects input shape (B, T, D) and adds positional encodings to the last dim.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.size()
        device = x.device
        position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / D))
        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)
        return self.dropout(x)


class AttnGainNet(nn.Module):
    """A tiny Transformer encoder for gain scheduling.

    Inputs: sequence of states (B, T, F)
    Output: gain multipliers (B, 3) mapped to safe ranges.
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        p_range: Tuple[float, float] = (0.6, 2.5),
        i_range: Tuple[float, float] = (0.5, 2.0),
        d_range: Tuple[float, float] = (0.5, 2.0),
    ) -> None:
        super().__init__()
        self.p_range = p_range
        self.i_range = i_range
        self.d_range = d_range

        self.input_proj = nn.Linear(state_dim, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 3)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, F)
        h = self.input_proj(x_seq)
        h = self.pos(h)
        h = self.encoder(h)  # (B, T, D)
        # use last token representation
        last = h[:, -1, :]
        raw = self.head(last)  # (B, 3)
        sig = torch.sigmoid(raw)
        p = self.p_range[0] + (self.p_range[1] - self.p_range[0]) * sig[..., 0]
        i = self.i_range[0] + (self.i_range[1] - self.i_range[0]) * sig[..., 1]
        d = self.d_range[0] + (self.d_range[1] - self.d_range[0]) * sig[..., 2]
        return torch.stack([p, i, d], dim=-1)


if __name__ == '__main__':
    B, T, F = 2, 8, 20
    net = AttnGainNet(state_dim=F)
    x = torch.randn(B, T, F)
    y = net(x)
    print(y.shape, y.min().item(), y.max().item())
