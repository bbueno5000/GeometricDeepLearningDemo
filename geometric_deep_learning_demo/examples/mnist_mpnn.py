"""
DOCSTRING
"""
import os
import torch
import torch_geometric

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'MNIST')
train_dataset = torch_geometric.datasets.MNISTSuperpixels(
    path, True, transform=torch_geometric.transforms.Cartesian())
test_dataset = torch_geometric.datasets.MNISTSuperpixels(
    path, False, transform=torch_geometric.transforms.Cartesian())
train_loader = torch_geometric.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
test_loader = torch_geometric.data.DataLoader(
    test_dataset, batch_size=64)
d = train_dataset.data

def normalized_cut_2d(edge_index, pos):
    """
    DOCSTRING
    """
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return torch_geometric.utils.normalized_cut(
        edge_index, edge_attr, num_nodes=pos.size(0))

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        n1 = torch.nn.Sequential(
            torch.nn.Linear(2, 25), torch.nn.ReLU(), torch.nn.Linear(25, 32))
        self.conv1 = torch_geometric.nn.NNConv(d.num_features, 32, n1)
        n2 = torch.nn.Sequential(
            torch.nn.Linear(2, 25), torch.nn.ReLU(), torch.nn.Linear(25, 2048))
        self.conv2 = torch_geometric.nn.NNConv(32, 64, n2)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, d.num_classes)

    def forward(self, data):
        """
        DOCSTRING
        """
        data.x = torch.nn.functional.elu(self.conv1(
            data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = torch_geometric.nn.graclus(
            data.edge_index, weight, data.x.size(0))
        data = torch_geometric.nn.max_pool(
            cluster, data, transform=torch_geometric.transforms.Cartesian(cat=False))
        data.x = torch.nn.functional.elu(self.conv2(
            data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = torch_geometric.nn.graclus(data.edge_index, weight, data.x.size(0))
        x, batch = torch_geometric.nn.max_pool_x(cluster, data.x, data.batch)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        return torch.nn.functional.log_softmax(self.fc2(x), dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def test():
    """
    DOCSTRING
    """
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)

def train(epoch):
    """
    DOCSTRING
    """
    model.train()
    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        torch.nn.functional.nll_loss(model(data), data.y).backward()
        optimizer.step()

if __name__ == '__main__':
    for epoch in range(30):
        train(epoch)
        test_acc = test()
        print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
