"""Losses for clean-room COMMAND training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from pyclad.video.models.command.architecture import CommandNetworkOutput


def command_loss(
    output: CommandNetworkOutput,
    labels: Tensor,
    bag_ids: Tensor | None = None,
    *,
    focal_gamma: float = 2.0,
    memory_weight: float = 0.25,
    margin_weight: float = 0.1,
    smoothness_weight: float = 0.01,
    separation_margin: float = 0.2,
) -> Tensor:
    """Combine weak-label, prototype-memory, and temporal objectives.

    ``labels`` may contain NaN for prediction-only or unlabeled windows. When
    no labels are available, the nearest-memory compactness objective still
    provides a differentiable anomaly-detection loss for pyCLAD strategies.
    """

    flat_labels = labels.reshape(output.logits.shape)
    valid = torch.isfinite(flat_labels)

    compactness = torch.minimum(output.normal_distance, output.anomaly_distance).mean()
    loss = memory_weight * compactness

    if valid.any():
        window_targets = flat_labels[valid].clamp(0.0, 1.0)
        window_logits = output.logits[valid]
        logits, targets = _bag_level_logits(
            window_logits,
            window_targets,
            None if bag_ids is None else bag_ids.reshape(output.logits.shape)[valid],
        )
        probabilities = torch.sigmoid(logits)
        binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_factor = torch.where(targets > 0.5, 1.0 - probabilities, probabilities).pow(focal_gamma)
        supervised = (focal_factor * binary_cross_entropy).mean()

        normal_distance = output.normal_distance[valid]
        anomaly_distance = output.anomaly_distance[valid]
        correct_distance = torch.where(window_targets > 0.5, anomaly_distance, normal_distance)
        incorrect_distance = torch.where(window_targets > 0.5, normal_distance, anomaly_distance)
        memory_fit = correct_distance.mean()
        separation = F.relu(separation_margin + correct_distance - incorrect_distance).mean()
        loss = supervised + memory_weight * memory_fit + margin_weight * separation

        positive_bags = logits[targets > 0.5]
        normal_bags = logits[targets <= 0.5]
        if len(positive_bags) and len(normal_bags):
            mil_margin = F.relu(1.0 - positive_bags.max() + normal_bags.max())
            loss = loss + margin_weight * mil_margin

    if output.logits.shape[-1] > 1:
        temporal_delta = output.logits[:, 1:] - output.logits[:, :-1]
        loss = loss + smoothness_weight * temporal_delta.square().mean()

    return loss


def _bag_level_logits(logits: Tensor, labels: Tensor, bag_ids: Tensor | None) -> tuple[Tensor, Tensor]:
    if bag_ids is None or not torch.isfinite(bag_ids).all():
        return logits, labels

    bag_logits = []
    bag_labels = []
    for bag_id in torch.unique(bag_ids):
        in_bag = bag_ids == bag_id
        bag_logits.append(logits[in_bag].max())
        bag_labels.append(labels[in_bag].max())
    return torch.stack(bag_logits), torch.stack(bag_labels)
