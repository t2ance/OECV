import numpy as np
from torch.nn import CrossEntropyLoss

np.float = float
import torch
from torch import nn


class NNClassifier(nn.Module):
    def __init__(self, n_feature: int, n_class: int, lr: float):
        super(NNClassifier, self).__init__()
        self.n_class = n_class
        self.n_feature = n_feature
        n_hidden = 32
        self.net = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_class)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        x = self.net(x)
        return x

    def partial_fit(self, datum, label):
        self.optimizer.zero_grad()
        outputs = self(datum)
        loss = self.criterion(outputs, label)
        loss.backward()
        self.optimizer.step()

    def fit(self, data, labels):
        batch_size = min(32, len(data))
        n_epoch = 100
        n_iter = len(data) // batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        criterion = CrossEntropyLoss()
        bar = range(n_epoch)
        for _ in bar:
            for i in range(n_iter):
                datum = data[i * batch_size:(i + 1) * batch_size]
                label = labels[i * batch_size:(i + 1) * batch_size]
                optimizer.zero_grad()
                out = self(datum)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

    def predict_proba(self, datum):
        with torch.no_grad():
            datum = torch.tensor(datum, dtype=torch.float32)
            return torch.softmax(self(datum), dim=1).numpy()

    def __str__(self):
        return str(self.to_dict())
