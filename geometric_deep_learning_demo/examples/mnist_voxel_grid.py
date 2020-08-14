"""
Voxel Grid Pooling 
"""
import os
import torch
import torch_geometric

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'MNIST')
transform = torch_geometric.transforms.Cartesian(cat=False)
train_dataset = torch_geometric.datasets.MNISTSuperpixels(path, True, transform=transform)
test_dataset = torch_geometric.datasets.MNISTSuperpixels(path, False, transform=transform)
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=64)

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch_geometric.nn.SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = torch_geometric.nn.SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = torch_geometric.nn.SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(4 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, data):
        """
        DOCSTRING
        """
        data.x = torch.nn.functional.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        cluster = torch_geometric.nn.voxel_grid(data.pos, data.batch, size=5, start=0, end=28)
        data = torch_geometric.nn.max_pool(cluster, data, transform=transform)
        data.x = torch.nn.functional.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        cluster = torch_geometric.nn.voxel_grid(data.pos, data.batch, size=7, start=0, end=28)
        data = torch_geometric.nn.max_pool(cluster, data, transform=transform)
        data.x = torch.nn.functional.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        cluster = torch_geometric.nn.voxel_grid(data.pos, data.batch, size=14, start=0, end=27.99)
        x = torch_geometric.nn.max_pool_x(cluster, data.x, data.batch, size=4)
        x = x.view(-1, self.fc1.weight.size(1))
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

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
    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        torch.nn.functional.nll_loss(model(data), data.y).backward()
        optimizer.step()

if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        test_acc = test()
        print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
