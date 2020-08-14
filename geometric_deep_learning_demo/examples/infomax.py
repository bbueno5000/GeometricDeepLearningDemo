"""
DOCSTRING
"""
import os
import torch
import torch_geometric

hidden_dim = 512
dataset = 'Cora'
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
data = torch_geometric.datasets.Planetoid(path, dataset)[0]

class Encoder(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.conv = torch_geometric.nn.GCNConv(data.num_features, hidden_dim)
        self.prelu = torch.nn.PReLU(hidden_dim)

    def forward(self, x, edge_index, corrupt=False):
        """
        DOCSTRING
        """
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = x[perm]
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

class Discriminator(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def forward(self, x, summary):
        """
        DOCSTRING
        """
        x = torch.matmul(x, torch.matmul(self.weight, summary))
        return x

    def reset_parameters(self):
        """
        DOCSTRING
        """
        size = self.weight.size(0)
        torch_geometric.nn.inits.uniform(size, self.weight)

class Infomax(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self, hidden_dim):
        super(Infomax, self).__init__()
        self.encoder = Encoder(hidden_dim)
        self.discriminator = Discriminator(hidden_dim)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        """
        DOCSTRING
        """
        positive = self.encoder(x, edge_index, corrupt=False)
        negative = self.encoder(x, edge_index, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))
        return l1 + l2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
infomax = Infomax(hidden_dim).to(device)
infomax_optimizer = torch.optim.Adam(infomax.parameters(), lr=0.001)

def train_infomax(epoch):
    """
    DOCSTRING
    """
    infomax.train()
    if epoch == 200:
        for param_group in infomax_optimizer.param_groups:
            param_group['lr'] = 0.0001
    infomax_optimizer.zero_grad()
    loss = infomax(data.x, data.edge_index)
    loss.backward()
    infomax_optimizer.step()
    return loss.item()

class Classifier(torch.nn.Module):
    """
    DOCSTRING
    """
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.lin = torch.nn.Linear(hidden_dim, data.num_classes)

    def forward(self, x, edge_index):
        """
        DOCSTRING
        """
        x = infomax.encoder(x, edge_index, corrupt=False)
        x = x.detach()
        x = self.lin(x)
        return torch.log_softmax(x, dim=-1)

    def reset_parameters(self):
        """
        DOCSTRING
        """
        self.lin.reset_parameters()

classifier = Classifier(hidden_dim).to(device)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

def test_classifier():
    """
    DOCSTRING
    """
    infomax.eval()
    classifier.eval()
    logits, accs = classifier(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train_classifier():
    """
    DOCSTRING
    """
    infomax.eval()
    classifier.train()
    classifier_optimizer.zero_grad()
    output = classifier(data.x, data.edge_index)
    loss = torch.nn.functional.nll_loss(
        output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    classifier_optimizer.step()
    return loss.item()

if __name__ == '__main__':
    print('Train deep graph infomax.')
    for epoch in range(300):
        loss = train_infomax(epoch)
        print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))
    print('Train logistic regression classifier.')
    for epoch in range(50):
        train_classifier()
        accs = test_classifier()
        log = 'Epoch: {:02d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *accs))
