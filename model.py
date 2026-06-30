import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class QKDAttackLocator(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(QKDAttackLocator, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = QKDAttackLocator(num_node_features=2, hidden_channels=16)
    print("GCN Architecture Initialized Successfully:")
    print(model)
