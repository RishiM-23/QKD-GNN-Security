import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class QKDAttackLocator(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(QKDAttackLocator, self).__init__()
        
        # Layer 1: Ingests raw telemetry features and expands to hidden dimensions
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        
        # Layer 2: Keeps hidden dimensions instead of condensing to 1
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # NEW: Edge Predictor Layer
        # Combines source and target node embeddings (hidden_channels * 2) into 1 edge output
        self.edge_predictor = nn.Linear(hidden_channels * 2, 1)

    def forward(self, data):
        # x represents node features (e.g., QBER, pulse count)
        # edge_index represents the adjacency matrix of the 4-node diamond topology
        x, edge_index = data.x, data.edge_index

        # First Message Passing Step
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Dropout layer to prevent overfitting on specific multi-hop paths
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second Message Passing Step
        x = self.conv2(x, edge_index)
        x = F.relu(x) # Add activation before pooling
        
        # NEW: Edge Pooling Step
        # Grab the source (src) and destination (dst) nodes for every edge
        src, dst = edge_index
        # Concatenate their embeddings to form edge features
        edge_embeddings = torch.cat([x[src], x[dst]], dim=-1)
        
        # Predict the attack probability for each specific edge
        edge_predictions = self.edge_predictor(edge_embeddings)

        # Sigmoid activation to output a probability (0.0 to 1.0) of being compromised
        return torch.sigmoid(edge_predictions)

from torch_geometric.nn import GCNConv

class QKDAttackLocator(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(QKDAttackLocator, self).__init__()
        
        # Layer 1: Ingests raw telemetry features and expands to hidden dimensions
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        
        # Layer 2: Condenses the hidden features down to a single binary output
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        # x represents node features (e.g., QBER, pulse count)
        # edge_index represents the adjacency matrix of the 4-node diamond topology
        x, edge_index = data.x, data.edge_index

        # First Message Passing Step
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Dropout layer to prevent overfitting on specific multi-hop paths
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second Message Passing Step
        x = self.conv2(x, edge_index)

        # Sigmoid activation to output a probability (0.0 to 1.0) of being compromised
        return torch.sigmoid(x)

if __name__ == "__main__":
    # Mock parameters: 2 input features (QBER, pulse count), 16 hidden dimensions
    model = QKDAttackLocator(num_node_features=2, hidden_channels=16)
    print("GCN Architecture Initialized Successfully:")
    print(model)
