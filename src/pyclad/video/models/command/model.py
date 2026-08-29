"""PyCLAD backbone for the strategy-compatible COMMAND model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.output.prediction_results import PredictionResults
from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.models.command.architecture import CommandNetwork
from pyclad.video.models.command.losses import command_loss


class CommandVideoModel(TorchBackbone):
    """COMMAND backbone for pyCLAD's existing PyTorch strategies.

    A strategy-facing matrix contains feature columns and, optionally, target
    columns described by ``strategy_schema``. Strategies that require the
    regular :class:`~pyclad.models.model.Model` interface can use pyCLAD's
    existing ``TorchModelAdapter``.
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
        threshold: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay must be non-negative")

        self.feature_dim = int(feature_dim)
        self.strategy_schema = strategy_schema or VideoStrategySchema(feature_dim=feature_dim)
        if self.strategy_schema.feature_dim != self.feature_dim:
            raise ValueError("strategy_schema.feature_dim must equal feature_dim")

        self._weak_label_index = self._target_index(weak_label_name)
        self._bag_id_index = self._target_index(bag_id_name)
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
        self.threshold = float(threshold)
        self._device = torch.device(device)
        self.module.to(self._device)

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
        return command_loss(self.module(features), labels, bag_ids)

    def forward(self, x: Tensor) -> Tensor:
        features, _, _ = self._split_tensor(x)
        return self.module(features).logits.reshape(features.shape[0], -1)

    def predict(self, data: np.ndarray) -> PredictionResults:
        matrix = self._validate_matrix(data, allow_features_only=True)
        self.module.eval()
        with torch.no_grad():
            batch = torch.as_tensor(matrix, dtype=torch.float32, device=self._device)
            anomaly_scores = torch.sigmoid(self.forward(batch)).reshape(-1).cpu().numpy()

        anomaly_scores = anomaly_scores.astype(np.float64, copy=False)
        return PredictionResults(
            y_pred=(anomaly_scores >= self.threshold).astype(np.int64),
            anomaly_scores=anomaly_scores,
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
            "device": str(self._device),
            "architecture": {
                "hidden_dim": self.module.feature_fusion.raw_projection.out_features,
                "embedding_dim": self.module.embedding[-1].out_features,
                "memory_size": self.module.memory.normal_memory.shape[0],
            },
        }

    def _target_index(self, name: str) -> Optional[int]:
        if name not in self.strategy_schema.target_names:
            return None
        return self.strategy_schema.target_names.index(name)

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
            labels = matrix[..., self.feature_dim + self._weak_label_index]
        if self._bag_id_index is not None and matrix.shape[-1] == self.strategy_schema.matrix_width:
            bag_ids = matrix[..., self.feature_dim + self._bag_id_index]
        return features, labels, bag_ids
