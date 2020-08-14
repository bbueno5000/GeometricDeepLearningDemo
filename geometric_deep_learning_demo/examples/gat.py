"""
Graph Attention Networks 
"""
import os
import torch
import torch_geometric

dataset = 'Cora'
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
data = torch_geometric.datasets.Planetoid(
    path, dataset, torch_geometric.transforms.NormalizeFeatures())[0]

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        self.att1 = torch_geometric.nn.GATConv(data.num_features, 8, heads=8, dropout=0.6)
        self.att2 = torch_geometric.nn.GATConv(8 * 8, data.num_classes, dropout=0.6)

    def forward(self):
        """
        DOCSTRING
        """
        x = torch.nn.functional.dropout(data.x, p=0.6, training=self.training)
        x = torch.nn.functional.elu(self.att1(x, data.edge_index))
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.att2(x, data.edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def test():
    """
    DOCSTRING
    """
    model.eval()
    logits, accs = model(), list()
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train():
    """
    DOCSTRING
    """
    model.train()
    optimizer.zero_grad()
    torch.nn.functional.nll_loss(
        model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

if __name__ == '__main__':
    for epoch in range(200):
        train()
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *test()))
