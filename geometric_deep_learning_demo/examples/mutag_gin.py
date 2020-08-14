"""
DOCSTRING
"""
import os
import torch
import torch_geometric

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'MUTAG')
dataset = torch_geometric.datasets.TUDataset(path, name='MUTAG').shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=128)
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=128)

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        num_features = dataset.num_features
        dim = 32
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.conv1 = torch_geometric.nn.GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.conv2 = torch_geometric.nn.GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.conv3 = torch_geometric.nn.GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        nn4 = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.conv4 = torch_geometric.nn.GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        nn5 = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.conv5 = torch_geometric.nn.GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        """
        DOCSTRING
        """
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = torch.nn.functional.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = torch.nn.functional.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = torch.nn.functional.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def test(loader):
    """
    DOCSTRING
    """
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def train(epoch):
    """
    DOCSTRING
    """
    model.train()
    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.functional.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)

if __name__ == '__main__':
    for epoch in range(100):
        train_loss = train(epoch)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        log = 'Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Acc: {:.7f}'
        print(log.format(epoch, train_loss, train_acc, test_acc))
