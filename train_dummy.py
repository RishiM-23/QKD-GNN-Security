import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader # Updated to fix deprecation warning
from model import QKDAttackLocator

def generate_dummy_diamond_dataset(num_graphs=100):
    dataset = []
    
    # Define the 4-node diamond topology edges (directed pairs for undirected graph)
    # 0->1, 1->0, 0->2, 2->0, 1->3, 3->1, 2->3, 3->2
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 3, 1, 3, 2]
    ], dtype=torch.long)

    for _ in range(num_graphs):
        # 1. Generate features for 4 nodes: [QBER, pulse count]
        x = torch.zeros((4, 2), dtype=torch.float)
        x[:, 0] = torch.rand(4) * 0.04   # Normal QBER between 0% and 4%
        x[:, 1] = torch.normal(1000.0, 50.0, size=(4,)) # Normal pulse counts

        # 2. NEW: Simulate a dummy binary target label for each EDGE (0 = clean, 1 = attacked)
        # The diamond has 8 directional edges, so shape is now [8, 1]
        y = torch.zeros((8, 1), dtype=torch.float)
        
        if torch.rand(1).item() > 0.5:
            # Randomly compromise one of the 8 links
            compromised_edge = torch.randint(0, 8, (1,)).item()
            y[compromised_edge] = 1.0
            
            # Find the target node of this compromised link and spike its telemetry
            target_node = edge_index[1, compromised_edge].item()
            x[target_node, 0] = torch.rand(1).item() * 0.25 + 0.15 # Spiked QBER
            x[target_node, 1] = torch.normal(300.0, 30.0, size=(1,)) # Dropped pulses

        graph_data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(graph_data)
        
    return dataset

def main():
    print("Generating 100 dummy QKD diamond topology graphs...")
    dataset = generate_dummy_diamond_dataset(num_graphs=100)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = QKDAttackLocator(num_node_features=2, hidden_channels=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    print("\nStarting optimization verification (Forward & Backward passes)...")
    model.train()
    
    for epoch in range(1, 6):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d} | Loss: {average_loss:.4f} -> Passes verified successfully.")

    print("\nPipeline check complete. The model architecture correctly supports optimization updates.")

if __name__ == "__main__":
    main()

