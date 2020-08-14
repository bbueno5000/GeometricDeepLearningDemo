"""
Top-K Pooling
"""
import os
import torch
import torch_geometric

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'ENZYMES')
dataset = torch_geometric.datasets.TUDataset(path, name='ENZYMES')
dataset = dataset.shuffle()
n = len(dataset) // 10
test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=60)
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=60)

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch_geometric.nn.GraphConv(dataset.num_features, 128)
        self.pool1 = torch_geometric.nn.TopKPooling(128, ratio=0.8)
        self.conv2 = torch_geometric.nn.GraphConv(128, 128)
        self.pool2 = torch_geometric.nn.TopKPooling(128, ratio=0.8)
        self.conv3 = torch_geometric.nn.GraphConv(128, 128)
        self.pool3 = torch_geometric.nn.TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        """
        DOCSTRING
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([
            torch_geometric.nn.global_max_pool(x, batch),
            torch_geometric.nn.global_mean_pool(x, batch)], dim=1)
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([
            torch_geometric.nn.global_max_pool(x, batch),
            torch_geometric.nn.global_mean_pool(x, batch)], dim=1)
        x = torch.nn.functional.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([
            torch_geometric.nn.global_max_pool(x, batch),
            torch_geometric.nn.global_mean_pool(x, batch)], dim=1)
        x = x1 + x2 + x3
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu(self.lin2(x))
        x = torch.nn.functional.log_softmax(self.lin3(x), dim=-1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def test(loader):
    """
    DOCSTRING
    """
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
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
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

if __name__ == '__main__':
    for epoch in range(200):
        loss = train(epoch)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        log = 'Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'
        print(log.format(epoch, loss, train_acc, test_acc))
