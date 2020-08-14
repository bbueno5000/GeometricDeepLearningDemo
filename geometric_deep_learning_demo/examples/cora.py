"""
DOCSTRING
"""
import os
import torch
import torch_geometric

dataset = 'Cora'
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
data = torch_geometric.datasets.Planetoid(
    path, dataset, torch_geometric.transforms.TargetIndegree())[0]
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.train_mask[:data.num_nodes - 1000] = 1
data.val_mask = None
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.test_mask[data.num_nodes - 500:] = 1

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch_geometric.nn.SplineConv(
            data.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = torch_geometric.nn.SplineConv(
            16, data.num_classes, dim=1, kernel_size=2)

    def forward(self):
        """
        DOCSTRING
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.nn.functional.dropout(x, training=self.training)
        x = torch.nn.functional.elu(self.conv1(x, edge_index, edge_attr))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return torch.nn.functional.log_softmax(x, dim=1)

device = torch.device('cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

def test():
    """
    DOCSTRING
    """
    model.eval()
    logits, accs = model(), list()
    for _, mask in data('train_mask', 'test_mask'):
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
        log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *test()))
