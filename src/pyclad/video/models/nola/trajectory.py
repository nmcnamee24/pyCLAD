"""Optional modern PyTorch trajectory predictor for NOLA."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class _NolaTrajectoryNetwork(nn.Module):
    def __init__(self, coordinate_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=coordinate_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, coordinate_dim)

    def forward(self, trajectories: Tensor) -> Tensor:
        sequence, _ = self.recurrent(trajectories)
        return self.output(sequence[:, -1])


class NolaTrajectoryPredictor:
    """Predict the next bounding box and return Euclidean track errors."""

    def __init__(
        self,
        *,
        coordinate_dim: int = 4,
        hidden_dim: int = 20,
        layers: int = 3,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 72,
        device: str | torch.device = "cpu",
    ):
        if min(coordinate_dim, hidden_dim, layers, epochs, batch_size) <= 0:
            raise ValueError("trajectory dimensions, layers, epochs, and batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.coordinate_dim = int(coordinate_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.module = _NolaTrajectoryNetwork(coordinate_dim, hidden_dim, layers).to(self.device)
        self._fit_calls = 0

    def fit(self, trajectories: np.ndarray, next_boxes: np.ndarray):
        x, y = self._arrays(trajectories, next_boxes)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        self.module.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                prediction = self.module(batch_x.to(self.device))
                loss = criterion(prediction, batch_y.to(self.device))
                loss.backward()
                optimizer.step()
        self._fit_calls += 1
        return self

    def predict(self, trajectories: np.ndarray) -> np.ndarray:
        if self._fit_calls == 0:
            raise RuntimeError("NOLA trajectory predictor must be fitted before prediction")
        x = np.asarray(trajectories, dtype=np.float32)
        self._validate_trajectories(x)
        self.module.eval()
        predictions = []
        with torch.no_grad():
            for offset in range(0, len(x), self.batch_size):
                batch = torch.from_numpy(x[offset : offset + self.batch_size]).to(self.device)
                predictions.append(self.module(batch).cpu().numpy())
        return np.concatenate(predictions) if predictions else np.empty((0, self.coordinate_dim), dtype=np.float32)

    def errors(self, trajectories: np.ndarray, next_boxes: np.ndarray) -> np.ndarray:
        _, targets = self._arrays(trajectories, next_boxes)
        return np.linalg.norm(self.predict(trajectories) - targets, axis=1)

    def additional_info(self) -> Dict[str, Any]:
        return {
            "coordinate_dim": self.coordinate_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "fit_calls": self._fit_calls,
        }

    def _arrays(self, trajectories: np.ndarray, next_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(trajectories, dtype=np.float32)
        y = np.asarray(next_boxes, dtype=np.float32)
        self._validate_trajectories(x)
        if y.ndim != 2 or y.shape != (len(x), self.coordinate_dim):
            raise ValueError(f"next_boxes must have shape ({len(x)}, {self.coordinate_dim}), got {y.shape}")
        if not np.isfinite(y).all():
            raise ValueError("next_boxes must contain only finite values")
        return x, y

    def _validate_trajectories(self, trajectories: np.ndarray) -> None:
        if trajectories.ndim != 3 or trajectories.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"trajectories must have shape (rows, time, {self.coordinate_dim}), got {trajectories.shape}"
            )
        if not np.isfinite(trajectories).all():
            raise ValueError("trajectories must contain only finite values")
