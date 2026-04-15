import numpy as np
import torch
from torch import nn

from pyclad.models.model import Model


class TinyDistillModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)
        self.train_loss = nn.MSELoss()
        self.lr = 1e-2

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TinyTorchModel(Model):
    def __init__(self):
        self.module = TinyDistillModule()
        self.epochs = 1
        self.batch_size = 2
        self.device = torch.device("cpu")
        self.fit_calls = 0

    def fit(self, data: np.ndarray):
        del data
        self.fit_calls += 1

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "TinyTorchModel"


class NonTorchModel(Model):
    def __init__(self):
        self.fit_calls = 0

    def fit(self, data: np.ndarray):
        del data
        self.fit_calls += 1

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "NonTorchModel"
