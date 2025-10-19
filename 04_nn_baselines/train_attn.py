"""train_attn.py
Supervised training for AttnGainNet using the same dataset as GSN.
We form short sequences by a sliding window over the flat (states, gains) pairs.
"""
import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pathlib, sys
_CUR = pathlib.Path(__file__).resolve().parent
if str(_CUR) not in sys.path:
    sys.path.insert(0, str(_CUR))
from attn_model import AttnGainNet  # type: ignore


class SeqDataset(Dataset):
    def __init__(self, npz_path: str, seq_len: int = 8, stride: int = 1):
        data = np.load(npz_path, allow_pickle=True)
        self.states = data['states'].astype(np.float32)
        self.gains = data['gains'].astype(np.float32)
        self.meta_raw = data['meta'][0]
        self.state_dim = self.states.shape[1]
        self.seq_len = max(2, int(seq_len))
        self.stride = max(1, int(stride))
        # prepare indices
        self.indices = []
        N = len(self.states)
        i = 0
        while i + self.seq_len <= N:
            self.indices.append(i)
            i += self.stride

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.seq_len
        x = self.states[s:e]
        # target gains use the last step's label
        y = self.gains[e - 1]
        return x, y


@dataclass
class TrainCfg:
    data: str
    seq_len: int = 8
    stride: int = 1
    epochs: int = 15
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    val_split: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir: str = '04_nn_baselines/results/checkpoints'


def _auto_find_dataset(path_hint: Optional[str]) -> str:
    if path_hint:
        if os.path.isfile(path_hint):
            print(f"[Info] 使用显式数据路径: {path_hint}")
            return path_hint
        raise FileNotFoundError(f"--data 提供的路径不存在: {path_hint}")
    cand = os.path.join('04_nn_baselines','data','gsn_dataset.npz')
    if os.path.isfile(cand):
        print(f"[Auto] 使用数据集: {cand}")
        return cand
    raise FileNotFoundError("未找到数据集，先运行 04_nn_baselines/dataset_builder.py 采集")


def train(cfg: TrainCfg):
    ds = SeqDataset(cfg.data, seq_len=cfg.seq_len, stride=cfg.stride)
    N = len(ds)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1 - cfg.val_split))
    train_idx, val_idx = idx[:split], idx[split:]

    def subset(indices):
        class _Sub(Dataset):
            def __init__(self, base, ids):
                self.base = base; self.ids = ids
            def __len__(self): return len(self.ids)
            def __getitem__(self, i):
                s = self.ids[i]
                x, y = base[s]
                return x, y
        base = ds
        return _Sub(ds, indices)

    train_loader = DataLoader(subset(train_idx), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(subset(val_idx), batch_size=cfg.batch_size, shuffle=False)

    model = AttnGainNet(state_dim=ds.state_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_val = float('inf')
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, 'attn_best.pt')

    for ep in range(1, cfg.epochs + 1):
        model.train()
        sum_loss = 0.0; cnt = 0
        for xs, ys in train_loader:
            xs = xs.to(cfg.device)
            ys = ys.to(cfg.device)
            preds = model(xs)
            loss = criterion(preds, ys)
            opt.zero_grad(); loss.backward(); opt.step()
            sum_loss += loss.item() * xs.size(0)
            cnt += xs.size(0)
        tr_loss = sum_loss / max(1, cnt)

        model.eval(); v_sum = 0.0; v_cnt = 0
        with torch.no_grad():
            for xs, ys in val_loader:
                xs = xs.to(cfg.device); ys = ys.to(cfg.device)
                preds = model(xs)
                loss = criterion(preds, ys)
                v_sum += loss.item() * xs.size(0)
                v_cnt += xs.size(0)
        va_loss = v_sum / max(1, v_cnt)
        print(f"[Epoch {ep:03d}] train_loss={tr_loss:.5f} val_loss={va_loss:.5f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({'model': model.state_dict(), 'val_loss': best_val, 'state_dim': ds.state_dim}, ckpt_path)
            print(f"  -> 保存最佳模型: {ckpt_path} (val_loss={best_val:.5f})")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default=None)
    ap.add_argument('--seq_len', type=int, default=8)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--ckpt_dir', type=str, default='04_nn_baselines/results/checkpoints')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = _auto_find_dataset(args.data)
    cfg = TrainCfg(data=data_path, seq_len=args.seq_len, stride=args.stride,
                   epochs=args.epochs, batch_size=args.bs, lr=args.lr,
                   weight_decay=args.weight_decay, val_split=args.val_split,
                   ckpt_dir=args.ckpt_dir)
    train(cfg)
