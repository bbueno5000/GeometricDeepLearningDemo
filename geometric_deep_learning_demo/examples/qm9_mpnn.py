"""
DOCSTRING
"""
import os
import torch
import torch_geometric

target = 0
dim = 73

class Transform:
    """
    DOCSTRING
    """
    def __call__(self, data):
        # pad features
        x = data.x
        data.x = torch.cat([x, x.new_zeros(x.size(0), dim - x.size(1))], dim=1)
        # specify target
        data.y = data.y[:, target]
        return data

class Complete:
    """
    DOCSTRING
    """
    def __call__(self, data):
        device = data.edge_index.device
        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr
        edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
            edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        return data

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'QM9')
transform = T.Compose([Transform(), Complete(), torch_geometric.transforms.Distance()])
dataset = torch_geometric.datasets.QM9(path, transform=transform).shuffle()
# normalize targets to mean = 0 and std = 1
mean = dataset.data.y[:, target].mean().item()
std = dataset.data.y[:, target].std().item()
dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std
# split datasets
test_dataset = dataset[:10000]
val_dataset = dataset[10000:20000]
train_dataset = dataset[20000:]
test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=64)
val_loader = torch_geometric.data.DataLoader(val_dataset, batch_size=64)
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

class Net(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self):
        super(Net, self).__init__()
        nn = torch.nn.Sequential(
            torch.nn.Linear(5, 128), torch.nn.ReLU(), torch.nn.Linear(128, dim * dim))
        self.conv = torch_geometric.nn.NNConv(dim, dim, nn, root_weight=False)
        self.gru = torch.nn.GRU(dim, dim, batch_first=True)
        self.set2set = torch_geometric.nn.Set2Set(dim, dim, processing_steps=3)
        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        """
        DOCSTRING
        """
        out = data.x
        h = data.x.unsqueeze(0)
        for i in range(3):
            m = torch.nn.functional.relu(
                self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(1), h)
            out = out.squeeze(1)
        out = self.set2set(out, data.batch)
        out = torch.nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(-1)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

def test(loader):
    """
    DOCSTRING
    """
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item() # MAE
    return error / len(loader.dataset)

def train(epoch):
    """
    DOCSTRING
    """
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

if __name__ == '__main__':
    best_val_error = None
    for epoch in range(300):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_error = test(val_loader)
        scheduler.step(val_error)
        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error
        log = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, Test MAE: {:.7f},'
        print(log.format(epoch, lr, loss, val_error, test_error))
