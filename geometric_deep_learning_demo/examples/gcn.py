"""
Graph Convolutional Networks 
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
        self.conv1 = torch_geometric.nn.GCNConv(data.num_features, 16, improved=False)
        self.conv2 = torch_geometric.nn.GCNConv(16, data.num_classes, improved=False)
        #self.conv1 = torch_geometric.nn.ChebConv(data.num_features, 16, K=2)
        #self.conv2 = torch_geometric.nn.ChebConv(16, data.num_features, K=2)

    def forward(self):
        """
        DOCSTRING
        """
        x, edge_index = data.x, data.edge_index
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

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
    torch.nn.functional.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

if __name__ == '__main__':
    best_val_acc = test_acc = 0
    for epoch in range(100):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
