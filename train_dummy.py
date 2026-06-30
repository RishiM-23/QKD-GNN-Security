import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import QKDAttackLocator  # Importing your working model

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
        # Baseline normal operation: Low QBER (~0.02), High pulse counts (~1000)
        x = torch.zeros((4, 2), dtype=torch.float)
        x[:, 0] = torch.rand(4) * 0.04   # Normal QBER between 0% and 4%
        x[:, 1] = torch.normal(1000.0, 50.0, size=(4,)) # Normal pulse counts around 1000

        # 2. Simulate a dummy binary target label for each node (0 = clean, 1 = attacked)
        # For this test, let's randomly decide if this graph has an attack
        y = torch.zeros((4, 1), dtype=torch.float)
        if torch.rand(1).item() > 0.5:
            # Randomly compromise one of the relay nodes (Node 1 or 2)
            compromised_node = torch.randint(1, 3, (1,)).item()
            y[compromised_node] = 1.0
            # Spike the QBER and drop the pulse count for the compromised node
            x[compromised_node, 0] = torch.rand(1).item() * 0.25 + 0.15 # 15%-40% QBER
            x[compromised_node, 1] = torch.normal(300.0, 30.0, size=(1,)) # Drastic drop in pulses

        # Create the PyG Data object
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(graph_data)
        
    return dataset

def main():
    # 1. Initialize Dataset and DataLoader
    print("Generating 100 dummy QKD diamond topology graphs...")
    dataset = generate_dummy_diamond_dataset(num_graphs=100)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 2. Instantiate Model, Loss, and Optimizer
    # 2 node features (QBER, pulses), 16 hidden channels
    model = QKDAttackLocator(num_node_features=2, hidden_channels=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss() # Using Binary Cross Entropy for standard classification baseline

    print("\nStarting optimization verification (Forward & Backward passes)...")
    model.train()
    
    # Run for 5 test epochs to verify gradients flow without crashing
    for epoch in range(1, 6):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()           # Clear out previous gradients
            out = model(batch)              # Forward pass
            loss = criterion(out, batch.y)  # Calculate classification loss
            loss.backward()                 # Backward pass (Verify backpropagation)
            optimizer.step()                # Update weights
            total_loss += loss.item() * batch.num_graphs

        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d} | Loss: {average_loss:.4f} -> Passes verified successfully.")

    print("\nPipeline check complete. The model architecture correctly supports optimization updates.")

if __name__ == "__main__":
    main()
