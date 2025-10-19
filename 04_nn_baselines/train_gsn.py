"""train_gsn.py
监督训练增益调度网络(GainSchedulingNet)：使用 dataset_builder 生成的 npz 数据。

Usage (示例):
    python -m nn_baselines.train_gsn --data results/gsn_dataset.npz --epochs 15 --bs 256 --lr 3e-4

输出:
    checkpoints/gsn_best.pt  (包含: model state_dict, meta)
    训练日志简单打印 (可后续扩展为TensorBoard)
"""
import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 允许脚本以两种方式运行:
# 1) python -m nn_baselines.train_gsn  (包上下文存在)
# 2) python nn_baselines/train_gsn.py  (无包上下文, 需要绝对导入)
# 导入策略：优先当前编号目录；兼容旧 nn_baselines 包名
import sys, pathlib
_CUR_DIR = pathlib.Path(__file__).resolve().parent
if str(_CUR_DIR) not in sys.path:
    sys.path.insert(0, str(_CUR_DIR))
try:
    from gsn_model import GainSchedulingNet  # 当前目录
except Exception as _e:
    raise ImportError("无法导入 GainSchedulingNet，请检查 04_nn_baselines/gsn_model.py 是否存在") from _e

# ================= Dataset =================
class GainDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.states = data['states'].astype(np.float32)
        self.gains = data['gains'].astype(np.float32)
        self.meta_raw = data['meta'][0]
        self.meta = {}
        try:
            self.meta = eval(self.meta_raw)  # 存储时为字符串
        except Exception:
            pass
        self.state_dim = self.states.shape[1]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.gains[idx]

# ================ Training Utilities ================
@dataclass
class TrainConfig:
    data: str
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    val_split: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ema_alpha: float = 0.0  # 若>0, 对标签做EMA平滑
    smooth_reg: float = 0.0 # 对输出相邻batch均值差分的正则(简化)
    # 统一保存到 nn_baselines/results/checkpoints
    ckpt_dir: str = 'nn_baselines/results/checkpoints'

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    @property
    def avg(self):
        return self.sum / max(1,self.cnt)
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n

# ================ Core Training ================

def train(cfg: TrainConfig):
    ds = GainDataset(cfg.data)
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
            def __getitem__(self, i): return base[self.ids[i]]
        base = ds
        return _Sub(ds, indices)

    train_loader = DataLoader(subset(train_idx), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(subset(val_idx), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = GainSchedulingNet(state_dim=ds.state_dim).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_val = float('inf')
    prev_train_mean = None

    for epoch in range(1, cfg.epochs+1):
        model.train()
        meter_train = AverageMeter()
        for states, gains in train_loader:
            states = states.to(cfg.device)
            gains = gains.to(cfg.device)
            preds = model(states)
            loss = criterion(preds, gains)
            if cfg.smooth_reg > 0 and prev_train_mean is not None:
                batch_mean = preds.mean(dim=0)
                smooth_pen = (batch_mean - prev_train_mean).pow(2).mean()
                loss = loss + cfg.smooth_reg * smooth_pen
                prev_train_mean = batch_mean.detach()
            else:
                prev_train_mean = preds.mean(dim=0).detach()
            optim.zero_grad(); loss.backward(); optim.step()
            meter_train.update(loss.item(), states.size(0))

        # 验证
        model.eval()
        meter_val = AverageMeter()
        with torch.no_grad():
            for states, gains in val_loader:
                states = states.to(cfg.device)
                gains = gains.to(cfg.device)
                preds = model(states)
                loss = criterion(preds, gains)
                meter_val.update(loss.item(), states.size(0))

        print(f"[Epoch {epoch:03d}] train_loss={meter_train.avg:.5f} val_loss={meter_val.avg:.5f}")

        if meter_val.avg < best_val:
            best_val = meter_val.avg
            ckpt_path = os.path.join(cfg.ckpt_dir, 'gsn_best.pt')
            torch.save({'model': model.state_dict(), 'val_loss': best_val, 'meta': ds.meta, 'state_dim': ds.state_dim}, ckpt_path)
            print(f"  -> 保存最佳模型: {ckpt_path} (val_loss={best_val:.5f})")

    print("训练完成。最佳验证损失:", best_val)

# ================ Entry Point ================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default=None, help='npz 数据路径 (可省略: 自动在 results/ 下寻找)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--smooth_reg', type=float, default=0.0)
    ap.add_argument('--ckpt_dir', type=str, default='04_nn_baselines/results/checkpoints', help='保存目录 (legacy: 04_nn_baselines/results/checkpoints)')
    return ap.parse_args()


def _auto_find_dataset(path_hint: Optional[str]) -> str:
    """解析/自动发现数据集路径。
    逻辑:
      1) 若显式提供 path_hint 且存在 -> 使用
      2) 若未提供: 优先 results/gsn_dataset.npz
      3) 若不存在: 扫描 results/*.npz
         - 若只发现 1 个 -> 使用
         - 若多个 -> 抛出错误要求显式指定
      4) 若仍未找到 -> 抛错
    """
    if path_hint:
        if os.path.isfile(path_hint):
            print(f"[Info] 使用显式数据路径: {path_hint}")
            return path_hint
        else:
            raise FileNotFoundError(f"--data 提供的路径不存在: {path_hint}")

    # 新默认路径优先
    # 新编号目录默认
    new_default = os.path.join('04_nn_baselines','data','gsn_dataset.npz')
    if os.path.isfile(new_default):
        print(f"[Auto] 使用新默认数据集: {new_default}")
        return new_default

    # 兼容旧路径
    legacy = os.path.join('results','gsn_dataset.npz')
    legacy_pkg = os.path.join('nn_baselines','data','gsn_dataset.npz')  # 旧包路径
    if os.path.isfile(legacy):
        print(f"[Auto][Legacy] 使用旧路径数据集: {legacy} (建议迁移到 04_nn_baselines/data/)")
        return legacy
    if os.path.isfile(legacy_pkg):
        print(f"[Auto][LegacyPkg] 使用旧包路径数据集: {legacy_pkg} (建议迁移到 04_nn_baselines/data/)")
        return legacy_pkg

    # 扫描两个目录
    scan_dirs = ['04_nn_baselines/data']
    candidates = []
    for d in scan_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.npz'):
                    candidates.append(os.path.join(d,f))
    if len(candidates) == 1:
        print(f"[Auto] 发现唯一 npz 数据集: {candidates[0]}")
        return candidates[0]
    elif len(candidates) == 0:
        raise FileNotFoundError("未提供 --data 且 nn_baselines/data 或 results/ 下没有任何 .npz 数据集。请先运行数据采集脚本。")
    else:
        raise RuntimeError("发现多个候选数据集: \n" + '\n'.join(candidates) + "\n请使用 --data 显式指定。")

if __name__ == '__main__':
    args = parse_args()
    data_path = _auto_find_dataset(args.data)
    cfg = TrainConfig(data=data_path, epochs=args.epochs, batch_size=args.bs, lr=args.lr,
                      weight_decay=args.weight_decay, val_split=args.val_split, smooth_reg=args.smooth_reg,
                      ckpt_dir=args.ckpt_dir)
    train(cfg)
