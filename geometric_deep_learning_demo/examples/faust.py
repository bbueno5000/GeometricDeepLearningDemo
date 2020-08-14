"""
DOCSTRING
"""
import os
import torch
import torch_geometric

class Transform:
    """
    DOCSTRING
    """
    def __call__(self, data):
        data.face, data.x = None, torch.ones(data.num_nodes, 1)
        return data

def norm(x, edge_index):
    """
    DOCSTRING
    """
    deg = torch_geometric.utils.degree(edge_index[0], x.size(0), x.dtype, x.device) + 1
    return x / deg.unsqueeze(-1)

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'FAUST')
pre_transform = torch_geometric.transforms.Compose(
    [torch_geometric.transforms.FaceToEdge(), Transform()])
train_dataset = torch_geometric.datasets.FAUST(
    path, True, torch_geometric.transforms.Cartesian(), pre_transform)
test_dataset = torch_geometric.datasets.FAUST(
    path, False, torch_geometric.transforms.Cartesian(), pre_transform)
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=1)
d = train_dataset[0]

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch_geometric.nn.SplineConv(1, 32, dim=3, kernel_size=5, norm=False)
        self.conv2 = torch_geometric.nn.SplineConv(32, 64, dim=3, kernel_size=5, norm=False)
        self.conv3 = torch_geometric.nn.SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.conv4 = torch_geometric.nn.SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.conv5 = torch_geometric.nn.SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.conv6 = torch_geometric.nn.SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.fc1 = torch.nn.Linear(64, 256)
        self.fc2 = torch.nn.Linear(256, d.num_nodes)

    def forward(self, data):
        """
        DOCSTRING
        """
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = torch.nn.functional.elu(norm(self.conv1(x, edge_index, pseudo), edge_index))
        x = torch.nn.functional.elu(norm(self.conv2(x, edge_index, pseudo), edge_index))
        x = torch.nn.functional.elu(norm(self.conv3(x, edge_index, pseudo), edge_index))
        x = torch.nn.functional.elu(norm(self.conv4(x, edge_index, pseudo), edge_index))
        x = torch.nn.functional.elu(norm(self.conv5(x, edge_index, pseudo), edge_index))
        x = torch.nn.functional.elu(norm(self.conv6(x, edge_index, pseudo), edge_index))
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def test():
    """
    DOCSTRING
    """
    model.eval()
    correct = 0
    for data in test_loader:
        pred = model(data.to(device)).max(1)[1]
        correct += pred.eq(target).sum().item()
    return correct / (len(test_dataset) * d.num_nodes)

def train(epoch):
    """
    DOCSTRING
    """
    model.train()
    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    for data in train_loader:
        optimizer.zero_grad()
        torch.nn.functional.nll_loss(model(data.to(device)), target).backward()
        optimizer.step()

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test_acc = test()
        print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
