"""
Dense Differentiable Pooling 
"""
import os
import torch
import torch_geometric

max_nodes = 100

class Filter:
    """
    DOCSTRING
    """
    def __call__(self, data):
        return data.num_nodes <= max_nodes

class Transform:
    """
    DOCSTRING
    """
    def __call__(self, data):
        data.x = data.x[:, :-3] # only use node attributes
        # add self loops
        arange = torch.arange(data.adj.size(-1), dtype=torch.long)
        data.adj[arange, arange] = 1
        return data

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'ENZYMES_d')
dataset = torch_geometric.datasets(
    path, name='ENZYMES',
    transform=torch_geometric.transforms.Compose(
        [torch_geometric.transforms.ToDense(max_nodes), Transform()]),
    pre_filter=Filter())
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = torch_geometric.data.DenseDataLoader(test_dataset, batch_size=20)
val_loader = torch_geometric.data.DenseDataLoader(val_dataset, batch_size=20)
train_loader = torch_geometric.data.DenseDataLoader(train_dataset, batch_size=20)

class GNN(torch.nn.Module):
    """
    Geometric Neural Network
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        lin=True,
        norm=True,
        norm_embed=True):
        super(GNN, self).__init__()
        self.conv1 = torch_geometric.nn.DenseSAGEConv(
            in_channels, hidden_channels, norm, norm_embed)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = torch_geometric.nn.DenseSAGEConv(
            hidden_channels, hidden_channels, norm, norm_embed)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = torch_geometric.nn.DenseSAGEConv(
            hidden_channels, out_channels, norm, norm_embed)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        if lin is True:
            self.lin = torch.nn.Linear(
                2 * hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        """
        DOCSTRING
        """
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj):
        """
        DOCSTRING
        """
        batch_size, num_nodes, in_channels = x.size()
        x0 = x
        x1 = self.bn(1, torch.nn.functional.relu(self.conv1(x0, adj)))
        x2 = self.bn(2, torch.nn.functional.relu(self.conv2(x1, adj)))
        x3 = self.bn(3, torch.nn.functional.relu(self.conv3(x2, adj)))
        x = torch.cat([x1, x2, x3], dim=-1)
        if self.lin is not None:
            x = torch.nn.functional.relu(self.lin(x))
        return x

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        self.gnn1_pool = GNN(18, 64, int(0.1 * max_nodes), norm=False)
        self.gnn1_embed = GNN(18, 64, 64, lin=False, norm=False)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False, norm=False)
        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, data):
        """
        DOCSTRING
        """
        x, adj = data.x, data.adj
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        x, adj, reg1 = torch_geometric.nn.dense_diff_pool(x, adj, s, data.mask)
        x = self.gnn2_embed(x, adj)
        x = x.mean(dim=1)
        x = torch.nn.functional.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1), reg1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def test(loader):
    """
    DOCSTRING
    """
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def train(epoch):
    """
    DOCSTRING
    """
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, reg = model(data)
        loss = torch.nn.functional.nll_loss(output, data.y.view(-1)) + reg
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

if __name__ == '__main__':
    best_val_acc = test_acc = 0
    for epoch in range(1000):
        train_loss = train(epoch)
        val_acc = test(val_loader)
        if val_acc > best_val_acc:
            test_acc = test(test_loader)
            best_val_acc = val_acc
        log = 'Epoch: {:03d}, Train Loss: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}'
        print(log.format(epoch, train_loss, val_acc, test_acc))
