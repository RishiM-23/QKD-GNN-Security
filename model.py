"""
hparam_tune.py  (Avyukth – Mon Jul 6)
Grid search over learning rate and hidden dimension.
Runs 5 quick epochs per config (enough to rank them), then re-trains
the winner for 50 epochs and saves the final frozen checkpoint.
"""
import sys, os
sys.path.insert(0, "/outputs")
sys.path.insert(0, "/home")

import torch, torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from train import load_graphs, TELEMETRY_CSV, SIGNAL_NORM, DECOY_NORM, EDGES_PER_GRAPH, NODE_COUNT, EDGE_INDEX
from key_rate_loss import CostSensitiveAttackLoss

DEVICE = torch.device("cpu")
BATCH  = 256
COST_W = 5.0
QUICK_EPOCHS = 5
FULL_EPOCHS  = 50

LR_GRID     = [1e-2, 1e-3, 5e-4]
HIDDEN_GRID = [32, 64, 128]

class QKD_GCN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(4, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1)
        )
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        e = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]], -1)
        return self.edge_head(e).squeeze(-1)

def run_quick(model, train_loader, val_loader, lr, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = CostSensitiveAttackLoss(COST_W)
    for _ in range(epochs):
        model.train()
        for b in train_loader:
            b = b.to(DEVICE); opt.zero_grad()
            loss = loss_fn(model(b), b.y, b.qber, b.signal_count, b.decoy_count)
            loss.backward(); opt.step()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in val_loader:
            b = b.to(DEVICE)
            preds  += (torch.sigmoid(model(b)) > 0.5).long().cpu().tolist()
            labels += b.y.long().cpu().tolist()
    return f1_score(labels, preds, zero_division=0)

def main():
    graphs = load_graphs(TELEMETRY_CSV)
    n_val  = int(0.2 * len(graphs))
    train_g, val_g = graphs[n_val:], graphs[:n_val]
    train_loader = DataLoader(train_g, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_g,   batch_size=BATCH, shuffle=False)

    print(f"{'LR':>8}  {'Hidden':>8}  {'Quick F1':>10}")
    print("─" * 32)
    results = []
    for lr in LR_GRID:
        for h in HIDDEN_GRID:
            model = QKD_GCN(hidden=h).to(DEVICE)
            f1 = run_quick(model, train_loader, val_loader, lr, QUICK_EPOCHS)
            print(f"{lr:>8.0e}  {h:>8}  {f1:>10.4f}")
            results.append((f1, lr, h))

    best_f1, best_lr, best_h = max(results)
    print(f"\nBest config: lr={best_lr}, hidden={best_h}  (quick F1={best_f1:.4f})")
    print(f"Re-training best config for {FULL_EPOCHS} epochs...")

    model = QKD_GCN(hidden=best_h).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = CostSensitiveAttackLoss(COST_W)
    best_val, best_state = 0.0, None
    for epoch in range(1, FULL_EPOCHS+1):
        model.train()
        for b in train_loader:
            b = b.to(DEVICE); opt.zero_grad()
            loss = loss_fn(model(b), b.y, b.qber, b.signal_count, b.decoy_count)
            loss.backward(); opt.step()
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for b in val_loader:
                b = b.to(DEVICE)
                preds  += (torch.sigmoid(model(b)) > 0.5).long().cpu().tolist()
                labels += b.y.long().cpu().tolist()
        val_f1 = f1_score(labels, preds, zero_division=0)
        sched.step(1 - val_f1)
        if val_f1 > best_val:
            best_val  = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}  Val F1: {val_f1:.4f}")

    torch.save(best_state, "gcn_tuned.pt")
    print(f"\nFinal best Val F1: {best_val:.4f}")
    print(f"Best hyperparams : lr={best_lr}, hidden={best_h}")
    print(f"Frozen checkpoint: gcn_tuned.pt")
    # Save config for Avyukth's freeze step
    import json
    with open("best_hparams.json", "w") as f:
        json.dump({"lr": best_lr, "hidden": best_h, "val_f1": best_val}, f, indent=2)

if __name__ == "__main__":
    main()