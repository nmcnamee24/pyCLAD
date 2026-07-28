"""pyCLAD model adapter for COMMAND."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.models.command.architecture import CommandNetwork
from pyclad.video.models.command.losses import command_loss
from pyclad.video.models.torch import TorchVideoBackbone
from pyclad.video.prediction_results import VideoPredictionResults


class CommandVideoModel(TorchVideoBackbone):
    """Weakly supervised COMMAND model usable by existing pyCLAD strategies.

    A strategy-facing matrix contains feature columns and, optionally, one
    ``weak_label`` target column described by ``strategy_schema``. Ordinary
    baseline and replay strategies call :meth:`fit`; EWC, LwF, A-GEM, and
    DER++ use the differentiable backbone methods directly.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        strategy_schema: Optional[VideoStrategySchema] = None,
        weak_label_name: str = "weak_label",
        bag_id_name: str = "bag_id",
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        memory_size: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 10,
        batch_size: int = 32,
        threshold: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay must be non-negative")

        self.feature_dim = int(feature_dim)
        self.strategy_schema = strategy_schema or VideoStrategySchema(feature_dim=feature_dim)
        if self.strategy_schema.feature_dim != self.feature_dim:
            raise ValueError("strategy_schema.feature_dim must equal feature_dim")
        if weak_label_name in self.strategy_schema.target_names:
            self._weak_label_index = self.strategy_schema.target_names.index(weak_label_name)
        else:
            self._weak_label_index = None
        if bag_id_name in self.strategy_schema.target_names:
            self._bag_id_index = self.strategy_schema.target_names.index(bag_id_name)
        else:
            self._bag_id_index = None

        self.weak_label_name = weak_label_name
        self.bag_id_name = bag_id_name
        self.module = CommandNetwork(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            memory_size=memory_size,
            dropout=dropout,
        )
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self._device = torch.device(device)
        self.module.to(self._device)
        self._fit_calls = 0

    def get_module(self) -> nn.Module:
        return self.module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.AdamW(
            self.module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def compute_loss(self, x: Tensor) -> Tensor:
        features, labels, bag_ids = self._split_tensor(x)
        output = self.module(features)
        return command_loss(output, labels, bag_ids)

    def forward(self, x: Tensor) -> Tensor:
        features, _, _ = self._split_tensor(x)
        return self.module(features).logits.reshape(features.shape[0], -1)

    def fit(self, data: np.ndarray):
        matrix = self._validate_matrix(data)
        if not len(matrix):
            raise ValueError("COMMAND cannot fit an empty feature matrix")
        loader = DataLoader(
            TensorDataset(torch.as_tensor(matrix, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.fit_with_loss(loader, lambda batch: self.compute_loss(batch[0].to(self._device)), self.epochs)
        self._fit_calls += 1
        return self

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        matrix = self._validate_matrix(data, allow_features_only=True)
        self.module.eval()
        scores = []
        with torch.no_grad():
            for offset in range(0, len(matrix), self.batch_size):
                batch = torch.as_tensor(
                    matrix[offset : offset + self.batch_size],
                    dtype=torch.float32,
                    device=self._device,
                )
                features, _, _ = self._split_tensor(batch)
                output = self.module(features)
                scores.append(torch.sigmoid(output.logits).reshape(-1).cpu().numpy())

        anomaly_scores = np.concatenate(scores).astype(np.float64, copy=False) if scores else np.empty(0)
        return VideoPredictionResults(
            y_pred=(anomaly_scores >= self.threshold).astype(np.int64),
            anomaly_scores=anomaly_scores,
            window_scores=anomaly_scores.copy(),
        )

    def name(self) -> str:
        return "COMMAND"

    def additional_info(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "weak_label_name": self.weak_label_name if self._weak_label_index is not None else None,
            "bag_id_name": self.bag_id_name if self._bag_id_index is not None else None,
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": str(self._device),
            "fit_calls": self._fit_calls,
            "architecture": {
                "hidden_dim": self.module.feature_fusion.raw_projection.out_features,
                "embedding_dim": self.module.embedding[-1].out_features,
                "memory_size": self.module.memory.normal_memory.shape[0],
            },
        }

    def _validate_matrix(self, data: np.ndarray, *, allow_features_only: bool = False) -> np.ndarray:
        matrix = np.asarray(data, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"COMMAND data must be two-dimensional, got {matrix.shape}")
        allowed_widths = {self.strategy_schema.matrix_width}
        if allow_features_only:
            allowed_widths.add(self.feature_dim)
        if matrix.shape[1] not in allowed_widths:
            expected = sorted(allowed_widths)
            raise ValueError(f"COMMAND data width must be one of {expected}, got {matrix.shape[1]}")
        return matrix

    def _split_tensor(self, matrix: Tensor) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        if matrix.ndim not in (2, 3):
            raise ValueError(f"COMMAND tensor must be two- or three-dimensional, got {tuple(matrix.shape)}")
        if matrix.shape[-1] not in (self.feature_dim, self.strategy_schema.matrix_width):
            raise ValueError(
                f"COMMAND tensor width must be {self.feature_dim} or "
                f"{self.strategy_schema.matrix_width}, got {matrix.shape[-1]}"
            )

        matrix = matrix.to(self._device)
        features = matrix[..., : self.feature_dim]
        target_shape = features.shape[:-1]
        labels = torch.full(target_shape, float("nan"), dtype=features.dtype, device=features.device)
        bag_ids = None
        if self._weak_label_index is not None and matrix.shape[-1] == self.strategy_schema.matrix_width:
            target_column = self.feature_dim + self._weak_label_index
            labels = matrix[..., target_column]
        if self._bag_id_index is not None and matrix.shape[-1] == self.strategy_schema.matrix_width:
            target_column = self.feature_dim + self._bag_id_index
            bag_ids = matrix[..., target_column]
        return features, labels, bag_ids
