"""
gcn_train.py  (Avyukth – Wed Jul 1)

GCN training loop with Monish's CostSensitiveAttackLoss integrated.
Loads the massive telemetry CSV (from Rishi's starter_v2.py), builds
per-epoch graph snapshots, trains a 2-layer GCN, and saves the best
checkpoint by validation F1 score.

Graph structure (matches tensor_serializer.cpp):
  Nodes : 4  (Node_A=0, Node_B=1, Node_C=2, Node_D=3)
  Edges : 4  (0→1, 0→2, 1→3, 2→3) -- diamond topology, fixed
  Edge features per graph snapshot:
    [qber, signal_count_norm, decoy_count_norm, key_loss]
  Node features: aggregated mean of adjacent edge features (since the
    GCN needs node-level features -- edges carry the telemetry signal)
  Labels: per-edge Attacked_Flag (binary)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from key_rate_loss import CostSensitiveAttackLoss

# ── Config ────────────────────────────────────────────────────────────────────
TELEMETRY_CSV = "sequence_telemetry_output.csv"
CHECKPOINT_PATH = "gcn_best.pt"
EDGES_PER_GRAPH = 4       # 4 links per epoch snapshot
NODE_COUNT = 4
EDGE_INDEX = torch.tensor([[0, 0, 1, 2],   # source nodes
                             [1, 2, 3, 3]], dtype=torch.long)  # dest nodes
SIGNAL_NORM = 10000.0
DECOY_NORM = 2000.0
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 256
COST_WEIGHT = 5.0
HIDDEN_DIM = 64
# ─────────────────────────────────────────────────────────────────────────────

def load_graphs(csv_path: str):
    df = pd.read_csv(csv_path)
    df["signal_norm"] = df["Signal_Count"] / SIGNAL_NORM
    df["decoy_norm"] = df["Decoy_Count"] / DECOY_NORM

    graphs = []
    num_epochs = len(df) // EDGES_PER_GRAPH

    for i in range(num_epochs):
        chunk = df.iloc[i * EDGES_PER_GRAPH: (i + 1) * EDGES_PER_GRAPH]

        # Edge features: [qber, signal_norm, decoy_norm, key_loss]
        edge_feat = torch.tensor(
            chunk[["QBER", "signal_norm", "decoy_norm", "Key_Loss"]].values,
            dtype=torch.float
        )

        # Node features: mean of incoming + outgoing edge features per node
        # Edge order: (A→B, A→C, B→D, C→D) → indices 0,1,2,3
        node_feat = torch.zeros(NODE_COUNT, edge_feat.shape[1])
        node_feat[0] += (edge_feat[0] + edge_feat[1]) / 2   # A: outgoing AB, AC
        node_feat[1] += (edge_feat[0] + edge_feat[2]) / 2   # B: AB in, BD out
        node_feat[2] += (edge_feat[1] + edge_feat[3]) / 2   # C: AC in, CD out
        node_feat[3] += (edge_feat[2] + edge_feat[3]) / 2   # D: incoming BD, CD

        labels = torch.tensor(chunk["Attacked_Flag"].values, dtype=torch.float)
        qber_t = torch.tensor(chunk["QBER"].values, dtype=torch.float)
        signal_t = torch.tensor(chunk["Signal_Count"].values, dtype=torch.float)
        decoy_t = torch.tensor(chunk["Decoy_Count"].values, dtype=torch.float)

        g = Data(
            x=node_feat,
            edge_index=EDGE_INDEX.clone(),
            edge_attr=edge_feat,
            y=labels,
            qber=qber_t,
            signal_count=signal_t,
            decoy_count=decoy_t,
        )
        graphs.append(g)

    print(f"Loaded {len(graphs)} graph snapshots from {csv_path}")
    return graphs


class QKD_GCN(nn.Module):
    def __init__(self, node_feat_dim: int = 4, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        # Edge-level attack classifier head:
        # concatenate the two endpoint node embeddings for each edge → 2*hidden
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),   # logit per edge
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Gather endpoint embeddings for each edge
        src_emb = x[edge_index[0]]   # (num_edges, hidden)
        dst_emb = x[edge_index[1]]   # (num_edges, hidden)
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)  # (num_edges, 2*hidden)
        return self.edge_head(edge_emb).squeeze(-1)        # (num_edges,)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(
            logits,
            batch.y,
            batch.qber,
            batch.signal_count,
            batch.decoy_count,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        labels = batch.y.long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return f1, all_preds, all_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    graphs = load_graphs(TELEMETRY_CSV)
    train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = QKD_GCN(node_feat_dim=4, hidden=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = CostSensitiveAttackLoss(cost_weight=COST_WEIGHT).to(device)

    best_f1 = 0.0
    print(f"\nTraining for {EPOCHS} epochs...\n{'─'*55}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val F1':>8}")
    print(f"{'─'*6}  {'─'*12}  {'─'*8}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_f1, val_preds, val_labels = evaluate(model, val_loader, device)
        scheduler.step(1 - val_f1)

        print(f"{epoch:>6}  {train_loss:>12.5f}  {val_f1:>8.4f}", end="")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  ← best", end="")
        print()

    print(f"\nBest Val F1: {best_f1:.4f}  |  Checkpoint: {CHECKPOINT_PATH}")
    print("\nFull classification report on validation set:")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    _, final_preds, final_labels = evaluate(model, val_loader, device)
    print(classification_report(final_labels, final_preds,
                                 target_names=["Clean", "Attacked"], zero_division=0))


if __name__ == "__main__":
    main()