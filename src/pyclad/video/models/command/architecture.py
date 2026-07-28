"""Clean-room COMMAND architecture for window-level video features.

The implementation follows the public architectural description of COMMAND:
augmented feature fusion, selective temporal modelling, and separate normal
and anomalous memories. It intentionally does not copy source from the
unlicensed reference repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class CommandNetworkOutput:
    """Per-window outputs used by training and inference."""

    logits: Tensor
    embeddings: Tensor
    normal_distance: Tensor
    anomaly_distance: Tensor


class AugmentedFeatureFusion(nn.Module):
    """Fuse raw, local-context, and temporal-change feature branches."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.raw_projection = nn.Linear(input_dim, hidden_dim)
        self.context_projection = nn.Linear(input_dim, hidden_dim)
        self.change_projection = nn.Linear(input_dim, hidden_dim)
        self.branch_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"COMMAND features must be 3D (batch, time, dim), got {tuple(features.shape)}")

        local_context = F.avg_pool1d(
            features.transpose(1, 2),
            kernel_size=3,
            stride=1,
            padding=1,
        ).transpose(1, 2)
        temporal_change = torch.diff(features, dim=1, prepend=features[:, :1])

        branches = torch.stack(
            [
                F.gelu(self.raw_projection(features)),
                F.gelu(self.context_projection(local_context)),
                F.gelu(self.change_projection(temporal_change)),
            ],
            dim=-2,
        )
        gates = torch.softmax(self.branch_gate(branches.flatten(start_dim=-2)), dim=-1)
        fused = (branches * gates.unsqueeze(-1)).sum(dim=-2)
        return self.output_norm(fused)


class SelectiveTemporalBlock(nn.Module):
    """A compact data-selective state-space block with a residual path."""

    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.content_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate_projection = nn.Linear(hidden_dim, hidden_dim)
        self.decay_projection = nn.Linear(hidden_dim, hidden_dim)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.input_norm(inputs)
        content = self.content_projection(normalized)
        content = self.depthwise_conv(content.transpose(1, 2)).transpose(1, 2)
        content = torch.tanh(content)
        gate = torch.sigmoid(self.gate_projection(normalized))
        decay = torch.sigmoid(self.decay_projection(normalized))

        state = torch.zeros_like(content[:, 0])
        outputs = []
        for step in range(content.shape[1]):
            step_decay = decay[:, step]
            state = step_decay * state + (1.0 - step_decay) * content[:, step]
            outputs.append(state * gate[:, step])

        sequence = torch.stack(outputs, dim=1)
        return inputs + self.output_projection(sequence)


class DualPrototypeMemory(nn.Module):
    """Trainable normal and anomalous prototype memories."""

    def __init__(self, embedding_dim: int, memory_size: int, temperature: float = 0.1):
        super().__init__()
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.normal_memory = nn.Parameter(torch.empty(memory_size, embedding_dim))
        self.anomaly_memory = nn.Parameter(torch.empty(memory_size, embedding_dim))
        self.temperature = float(temperature)
        nn.init.xavier_uniform_(self.normal_memory)
        nn.init.xavier_uniform_(self.anomaly_memory)

    def forward(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        normal_distance = self._soft_nearest_distance(embeddings, self.normal_memory)
        anomaly_distance = self._soft_nearest_distance(embeddings, self.anomaly_memory)
        return normal_distance, anomaly_distance

    def _soft_nearest_distance(self, embeddings: Tensor, memory: Tensor) -> Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=-1)
        normalized_memory = F.normalize(memory, dim=-1)
        cosine_distance = 1.0 - torch.einsum("btd,md->btm", normalized_embeddings, normalized_memory)
        weights = torch.softmax(-cosine_distance / self.temperature, dim=-1)
        return (weights * cosine_distance).sum(dim=-1)


class CommandNetwork(nn.Module):
    """COMMAND feature fusion, temporal modelling, and dual-memory scorer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        memory_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and embedding_dim must be positive")

        self.input_dim = int(input_dim)
        self.feature_fusion = AugmentedFeatureFusion(input_dim, hidden_dim)
        self.temporal = SelectiveTemporalBlock(hidden_dim)
        self.embedding = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.memory = DualPrototypeMemory(embedding_dim, memory_size)
        self.classifier = nn.Linear(embedding_dim, 1)
        self.memory_logit_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: Tensor) -> CommandNetworkOutput:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"COMMAND expects (rows, {self.input_dim}) or "
                f"(batch, time, {self.input_dim}), got {tuple(features.shape)}"
            )

        fused = self.feature_fusion(features)
        temporal = self.temporal(fused)
        embeddings = self.embedding(temporal)
        normal_distance, anomaly_distance = self.memory(embeddings)
        memory_logit = normal_distance - anomaly_distance
        logits = self.classifier(embeddings).squeeze(-1)
        logits = logits + self.memory_logit_weight * memory_logit
        return CommandNetworkOutput(
            logits=logits,
            embeddings=embeddings,
            normal_distance=normal_distance,
            anomaly_distance=anomaly_distance,
        )
