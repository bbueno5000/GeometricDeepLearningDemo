"""
Attention-based Graph Neural Network
"""
import os
import torch
import torch_geometric

dataset = 'Cora'
PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
data = torch_geometric.datasets.Planetoid(
    PATH, dataset, torch_geometric.transforms.NormalizeFeatures())[0]

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(data.num_features, 16)
        self.prop1 = torch_geometric.nn.AGNNProp(requires_grad=False)
        self.prop2 = torch_geometric.nn.AGNNProp(requires_grad=True)
        self.fc2 = torch.nn.Linear(16, data.num_classes)

    def forward(self):
        """
        DOCSTRING
        """
        x = torch.nn.functional.dropout(data.x, training=self.training)
        x = torch.nn.functionalF.relu(self.fc1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
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
    torch.nn.functional.nll_loss(
        model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

if __name__ == '__main__':
    for epoch in range(100):
        train()
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *test()))
