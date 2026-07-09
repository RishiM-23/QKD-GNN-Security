"""
evaluate.py

Compares the trained GCN (gcn_tuned.pt) against a network-wide QBER
threshold baseline on the repo's actual telemetry data.

Works with the repo as-is:
    sequence_telemetry_output.csv   <- Rishi's simulation output
    gcn_tuned.pt                    <- Avyukth's best checkpoint
    gcn_train.py                    <- defines QKD_GCN + load_graphs
    key_rate_loss.py                <- key-rate cost helpers

Usage:
    python evaluate.py
    python evaluate.py --checkpoint gcn_best.pt
    python evaluate.py --threshold-only
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# ── Reproduce QKD_GCN inline so evaluate.py has zero path-dependency ─────────
# (mirrors gcn_train.py exactly — keep in sync if Avyukth changes the arch)
class QKD_GCN(nn.Module):
    def __init__(self, node_feat_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x,      data.edge_index))
        src = x[data.edge_index[0]]
        dst = x[data.edge_index[1]]
        return self.edge_head(torch.cat([src, dst], dim=-1)).squeeze(-1)  # logits


# ── Graph construction (matches gcn_train.py load_graphs) ────────────────────
EDGES_PER_GRAPH = 4
NODE_COUNT      = 4
SIGNAL_NORM     = 10000.0
DECOY_NORM      = 2000.0
EDGE_INDEX      = torch.tensor([[0, 0, 1, 2],
                                 [1, 2, 3, 3]], dtype=torch.long)


def load_graphs(csv_path: str):
    df = pd.read_csv(csv_path)
    df["signal_norm"] = df["Signal_Count"] / SIGNAL_NORM
    df["decoy_norm"]  = df["Decoy_Count"]  / DECOY_NORM

    graphs, n = [], len(df) // EDGES_PER_GRAPH
    for i in range(n):
        chunk = df.iloc[i * EDGES_PER_GRAPH:(i + 1) * EDGES_PER_GRAPH]
        ef = torch.tensor(
            chunk[["QBER", "signal_norm", "decoy_norm", "Key_Loss"]].values,
            dtype=torch.float)
        nf = torch.zeros(NODE_COUNT, 4)
        nf[0] = (ef[0] + ef[1]) / 2
        nf[1] = (ef[0] + ef[2]) / 2
        nf[2] = (ef[1] + ef[3]) / 2
        nf[3] = (ef[2] + ef[3]) / 2
        graphs.append(Data(
            x=nf,
            edge_index=EDGE_INDEX.clone(),
            y=torch.tensor(chunk["Attacked_Flag"].values, dtype=torch.float),
            qber=torch.tensor(chunk["QBER"].values,          dtype=torch.float),
            signal_count=torch.tensor(chunk["Signal_Count"].values, dtype=torch.float),
            key_loss=torch.tensor(chunk["Key_Loss"].values,  dtype=torch.float),
        ))
    print(f"Loaded {len(graphs)} graph snapshots from {csv_path}")
    return graphs, df


# ── Key-rate cost (no extra import needed) ────────────────────────────────────
BASELINE_QBER = 0.02
Q_SIFT, F_EC, EPS = 0.5, 1.16, 1e-7


def _h2(x):
    x = np.clip(x, EPS, 1 - EPS)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def _skr(qber, signal):
    q = signal / SIGNAL_NORM
    return np.maximum(Q_SIFT * (-F_EC * q * _h2(qber) + q * (1 - _h2(qber))), 0.0)


def key_rate_cost(qber, signal):
    return np.maximum(_skr(np.full_like(qber, BASELINE_QBER), signal) - _skr(qber, signal), 0.0)


# ── Threshold baseline ────────────────────────────────────────────────────────
def threshold_predict(df: pd.DataFrame, t: float) -> np.ndarray:
    preds = []
    for _, g in df.groupby(df.index // EDGES_PER_GRAPH):
        flag = 1 if g["QBER"].mean() > t else 0
        preds.extend([flag] * EDGES_PER_GRAPH)
    return np.array(preds)


def best_threshold(df: pd.DataFrame) -> tuple:
    labels = df["Attacked_Flag"].values
    best_f1, best_t = 0.0, 0.05
    for t in np.arange(0.02, 0.25, 0.005):
        f1 = f1_score(labels, threshold_predict(df, t), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 4)
    return best_t, best_f1


# ── GCN inference ─────────────────────────────────────────────────────────────
def gcn_predict(graphs: list, checkpoint: str, hidden: int = 32) -> np.ndarray:
    state  = torch.load(checkpoint, map_location="cpu")
    hidden = state["conv1.bias"].shape[0]  # auto-detect from checkpoint
    model  = QKD_GCN(node_feat_dim=4, hidden=hidden)
    model.load_state_dict(state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=256, shuffle=False):
            logits = model(batch)
            preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().tolist())
    return np.array(preds)


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_report(title, labels, preds, qbers, signals, key_losses):
    p  = precision_score(labels, preds, zero_division=0)
    r  = recall_score(labels,    preds, zero_division=0)
    f1 = f1_score(labels,        preds, zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  Overall   Prec={p:.3f}  Rec={r:.3f}  F1={f1:.3f}")

    # Per attack type inferred from Key_Loss
    # (stealthy: 0.10-0.30, blatant: 0.80-1.00  per qkd_simulation.py)
    print(f"\n  {'Attack type':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N attacks':>10}")
    print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*6} {'─'*10}")
    for name, mask_fn in [
        ("Stealthy",  lambda kl: (labels == 1) & (kl <  0.5)),
        ("Blatant",   lambda kl: (labels == 1) & (kl >= 0.5)),
    ]:
        m = mask_fn(key_losses)
        if not m.any():
            continue
        # Isolate this attack type: zero out other attack type's labels
        tmp = labels.copy()
        other = (labels == 1) & ~m
        tmp[other] = 0
        tp = precision_score(tmp, preds, zero_division=0)
        tr = recall_score(tmp,    preds, zero_division=0)
        tf = f1_score(tmp,        preds, zero_division=0)
        print(f"  {name:<14} {tp:>6.3f} {tr:>6.3f} {tf:>6.3f} {int(m.sum()):>10}")

    # Key-rate cost
    tp_m = (labels == 1) & (preds == 1)
    fn_m = (labels == 1) & (preds == 0)
    saved  = key_rate_cost(qbers[tp_m], signals[tp_m]).sum() if tp_m.any() else 0.0
    missed = key_rate_cost(qbers[fn_m], signals[fn_m]).sum() if fn_m.any() else 0.0
    total  = saved + missed
    ratio  = saved / total if total > 0 else 0.0
    print(f"\n  Key-rate cost saved  (TP): {saved:>10.4f}")
    print(f"  Key-rate cost missed (FN): {missed:>10.4f}")
    print(f"  Cost capture ratio       : {ratio:>10.1%}")

    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "cost_saved": float(saved), "cost_missed": float(missed),
            "cost_capture_ratio": float(ratio)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main(telemetry_csv="sequence_telemetry_output.csv",
         checkpoint="gcn_tuned.pt",
         hidden=32,
         threshold_only=False):

    print("=" * 60)
    print("  QKD ATTACK DETECTION — EVALUATION REPORT")
    print("=" * 60)

    if not os.path.exists(telemetry_csv):
        print(f"\n[ERROR] {telemetry_csv} not found.")
        print("  Run: python qkd_simulation.py   to generate it first.")
        return

    graphs, df = load_graphs(telemetry_csv)

    # Reproduce the same 80/20 train/val split as gcn_train.py
    _, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    n_total = len(graphs)
    n_val   = len(val_graphs)
    # Match the val rows in the flat dataframe
    all_idx = list(range(n_total))
    _, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    val_rows = []
    for i in sorted(val_idx):
        val_rows.append(df.iloc[i * EDGES_PER_GRAPH:(i + 1) * EDGES_PER_GRAPH])
    val_df = pd.concat(val_rows).reset_index(drop=True)

    labels     = val_df["Attacked_Flag"].values
    qbers      = val_df["QBER"].values
    signals    = val_df["Signal_Count"].values
    key_losses = val_df["Key_Loss"].values

    summary = {}

    # 1. Threshold baseline
    best_t, best_t_f1 = best_threshold(val_df)
    print(f"\nBest threshold T = {best_t:.3f}  (F1 = {best_t_f1:.4f})")
    thresh_preds = threshold_predict(val_df, best_t)
    summary["threshold"] = print_report(
        f"THRESHOLD BASELINE  (T = {best_t})",
        labels, thresh_preds, qbers, signals, key_losses)
    summary["threshold"]["T"] = best_t

    # 2. GCN
    if not threshold_only:
        if not os.path.exists(checkpoint):
            print(f"\n[ERROR] Checkpoint not found: {checkpoint}")
            print("  Run: python gcn_train.py   to generate it, or")
            print("       python hparam_tune.py  for the tuned version.")
            return
        gcn_preds = gcn_predict(val_graphs, checkpoint, hidden=hidden)
        summary["gcn"] = print_report(
            f"GCN — QKD_GCN  (hidden={hidden}, checkpoint={checkpoint})",
            labels, gcn_preds, qbers, signals, key_losses)
        summary["gcn"]["checkpoint"] = checkpoint

        # GCN vs threshold summary
        gcn_f1  = summary["gcn"]["f1"]
        thr_f1  = summary["threshold"]["f1"]
        delta   = gcn_f1 - thr_f1
        print(f"\n{'═'*60}")
        print(f"  GCN vs Threshold  |  ΔF1 = {delta:+.4f}  "
              f"({'GCN wins' if delta > 0 else 'Threshold wins' if delta < 0 else 'Tied'})")
        print(f"{'═'*60}")

    with open("eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → eval_summary.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-csv", default="sequence_telemetry_output.csv")
    parser.add_argument("--checkpoint",    default="gcn_tuned.pt")
    parser.add_argument("--hidden",        type=int, default=32)
    parser.add_argument("--threshold-only", action="store_true")
    args = parser.parse_args()
    main(telemetry_csv=args.telemetry_csv,
         checkpoint=args.checkpoint,
         hidden=args.hidden,
         threshold_only=args.threshold_only)