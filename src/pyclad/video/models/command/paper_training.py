"""Bag-aware optimization for PyCLAD's COMMAND paper recreation.

The regular pyCLAD strategy interface intentionally operates on two-dimensional
row matrices.  COMMAND's paper architecture instead requires complete
32-snippet video bags, so this module owns its replay and training loop rather
than silently flattening temporal sequences through a generic strategy.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pyclad.video.data.sample import VideoWindow
from pyclad.video.data.video_concept import VideoConcept
from pyclad.video.models.command.paper_architecture import (
    PaperAugFuseNet,
    PaperCommandArchitectureConfig,
    PaperCommandNetwork,
    PaperCommandOutput,
    PaperTempMamba,
)


@dataclass(frozen=True)
class PaperCommandLossConfig:
    """Published composite weights plus explicitly documented paper-silent defaults."""

    mil_weight: float = 1.0
    contrastive_weight: float = 1.0
    focal_weight: float = 0.5
    anomaly_separation_weight: float = 0.3
    contrastive_temperature: float = 0.15
    mil_margin: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.5
    anomaly_margin: float = 1.0
    sparsity_weight: float = 8e-5
    smoothness_weight: float = 8e-5
    score_l2_weight: float = 5e-5

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if name == "focal_alpha":
                if not 0.0 <= value <= 1.0:
                    raise ValueError("focal_alpha must be in [0, 1]")
            elif value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.contrastive_temperature == 0.0:
            raise ValueError("contrastive_temperature must be positive")


@dataclass(frozen=True)
class PaperCommandTrainerConfig:
    """Optimization, replay, and reproducibility settings for ContTrain++."""

    architecture: PaperCommandArchitectureConfig = field(default_factory=PaperCommandArchitectureConfig)
    loss: PaperCommandLossConfig = field(default_factory=PaperCommandLossConfig)
    epochs: int = 100
    batch_size: int = 32
    replay_batch_size: int = 16
    buffer_size: int = 1_000
    learning_rate: float = 1e-4
    secondary_memory_learning_rate: float = 1e-5
    gradient_clip: float = 1.0
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    novelty_mad_scale: float = 3.0
    novelty_warmup_epochs: int = 1
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        integer_fields = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "lr_step_size": self.lr_step_size,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.replay_batch_size < self.batch_size:
            raise ValueError("replay_batch_size must be in [0, batch_size)")
        if self.learning_rate <= 0 or self.secondary_memory_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.gradient_clip <= 0 or self.lr_gamma <= 0:
            raise ValueError("gradient_clip and lr_gamma must be positive")
        if self.novelty_mad_scale < 0 or self.novelty_warmup_epochs < 0:
            raise ValueError("novelty controls must be non-negative")


@dataclass(frozen=True)
class PaperVideoBag:
    """One complete video bag retained by the paper trainer and replay buffer."""

    bag_id: str
    task_id: str
    features: np.ndarray
    weak_label: int
    windows: Tuple[VideoWindow, ...] = ()

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if not self.bag_id or not self.task_id:
            raise ValueError("bag_id and task_id must be non-empty")
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(f"paper COMMAND bags must have shape (time, features), got {features.shape}")
        if len(features) == 0 or not np.isfinite(features).all():
            raise ValueError("paper COMMAND bag features must be finite and non-empty")
        if self.weak_label not in {0, 1}:
            raise ValueError("weak_label must be zero or one")
        if self.windows and len(self.windows) != len(features):
            raise ValueError("windows and bag features must have the same temporal length")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "windows", tuple(self.windows))


@dataclass(frozen=True)
class ReplayEntry:
    bag: PaperVideoBag
    score_curve: np.ndarray

    def __post_init__(self) -> None:
        curve = np.asarray(self.score_curve, dtype=np.float32).reshape(-1)
        if curve.shape != (len(self.bag.features),) or not np.isfinite(curve).all():
            raise ValueError("replay score curve must be finite and match the bag length")
        object.__setattr__(self, "score_curve", curve)

    @property
    def maximum_score(self) -> float:
        return float(np.max(self.score_curve))

    @property
    def peak_quartile(self) -> int:
        length = len(self.score_curve)
        return min(3, int(np.argmax(self.score_curve)) * 4 // max(1, length))


class ContTrainReplayBuffer:
    """Deterministic, label/task/temporal-balanced complete-bag replay."""

    def __init__(self, max_size: int = 1_000):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self._entries: Dict[str, ReplayEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> Tuple[ReplayEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def add(self, entries: Iterable[ReplayEntry]) -> None:
        for entry in entries:
            self._entries[entry.bag.bag_id] = entry
        if len(self._entries) > self.max_size:
            retained = self._balanced_select(self.entries, self.max_size)
            self._entries = {entry.bag.bag_id: entry for entry in retained}

    def sample(self, count: int, *, exclude_bag_ids: Iterable[str] = ()) -> Tuple[PaperVideoBag, ...]:
        if count <= 0:
            return ()
        excluded = set(exclude_bag_ids)
        available = tuple(entry for entry in self.entries if entry.bag.bag_id not in excluded)
        return tuple(entry.bag for entry in self._balanced_select(available, min(count, len(available))))

    @classmethod
    def _balanced_select(cls, entries: Sequence[ReplayEntry], count: int) -> Tuple[ReplayEntry, ...]:
        if count <= 0 or not entries:
            return ()
        labels = sorted({entry.bag.weak_label for entry in entries})
        selected: list[ReplayEntry] = []
        if len(labels) == 2:
            budgets = {labels[0]: count // 2, labels[1]: count - count // 2}
        else:
            budgets = {labels[0]: count}
        for label in labels:
            candidates = [entry for entry in entries if entry.bag.weak_label == label]
            selected.extend(cls._round_robin_cells(candidates, min(budgets[label], len(candidates))))

        if len(selected) < count:
            used = {entry.bag.bag_id for entry in selected}
            remaining = [entry for entry in entries if entry.bag.bag_id not in used]
            remaining.sort(key=lambda item: (-item.maximum_score, item.bag.task_id, item.bag.bag_id))
            selected.extend(remaining[: count - len(selected)])
        return tuple(selected[:count])

    @staticmethod
    def _round_robin_cells(entries: Sequence[ReplayEntry], count: int) -> list[ReplayEntry]:
        cells: Dict[tuple[str, int], list[ReplayEntry]] = {}
        for entry in entries:
            cells.setdefault((entry.bag.task_id, entry.peak_quartile), []).append(entry)
        for values in cells.values():
            values.sort(key=lambda item: (-item.maximum_score, item.bag.bag_id))
        keys = sorted(cells)
        selected: list[ReplayEntry] = []
        offset = 0
        while len(selected) < count:
            added = False
            for key in keys:
                values = cells[key]
                if offset < len(values):
                    selected.append(values[offset])
                    added = True
                    if len(selected) == count:
                        break
            if not added:
                break
            offset += 1
        return selected

    def state_dict(self) -> Mapping[str, object]:
        return {
            "max_size": self.max_size,
            "entries": [
                {
                    "bag_id": entry.bag.bag_id,
                    "task_id": entry.bag.task_id,
                    "features": entry.bag.features,
                    "weak_label": entry.bag.weak_label,
                    "score_curve": entry.score_curve,
                }
                for entry in self.entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["max_size"]) != self.max_size:
            raise ValueError("checkpoint replay capacity does not match trainer configuration")
        self._entries.clear()
        for item in state["entries"]:
            bag = PaperVideoBag(
                bag_id=str(item["bag_id"]),
                task_id=str(item["task_id"]),
                features=np.asarray(item["features"], dtype=np.float32),
                weak_label=int(item["weak_label"]),
            )
            entry = ReplayEntry(bag=bag, score_curve=np.asarray(item["score_curve"], dtype=np.float32))
            self._entries[bag.bag_id] = entry


@dataclass
class PaperNormalOnlyOutput:
    temporal_features: Tensor
    nearest_distance: Tensor
    nearest_slot: Tensor


class PaperNormalOnlyNetwork(nn.Module):
    """Paper encoder with one normal bank and no binary anomaly classifier."""

    def __init__(self, config: PaperCommandArchitectureConfig):
        super().__init__()
        self.config = config
        self.feature_fusion = PaperAugFuseNet(config.appearance_dim, config.motion_dim)
        self.temporal = PaperTempMamba(config)
        self.normal_memory = nn.Parameter(torch.empty(config.memory_size, config.fused_dim))
        nn.init.normal_(self.normal_memory, mean=0.0, std=0.01)

    def forward(self, appearance: Tensor, motion: Tensor) -> PaperNormalOnlyOutput:
        temporal = self.temporal(self.feature_fusion(appearance, motion))
        distances = torch.cdist(temporal, self.normal_memory.unsqueeze(0))
        nearest_distance, nearest_slot = distances.min(dim=-1)
        return PaperNormalOnlyOutput(temporal, nearest_distance, nearest_slot)


class PaperCommandVideoModel(nn.Module):
    """High-level model that accepts only complete 2048-D video bags."""

    def __init__(
        self,
        architecture: PaperCommandArchitectureConfig | None = None,
        *,
        mode: Literal["dual-memory", "normal-only"] = "dual-memory",
    ):
        super().__init__()
        self.architecture = architecture or PaperCommandArchitectureConfig()
        if mode not in {"dual-memory", "normal-only"}:
            raise ValueError("mode must be 'dual-memory' or 'normal-only'")
        self.mode = mode
        self.network: PaperCommandNetwork | PaperNormalOnlyNetwork
        if mode == "dual-memory":
            self.network = PaperCommandNetwork(self.architecture)
        else:
            self.network = PaperNormalOnlyNetwork(self.architecture)

    def forward(self, bags: Tensor, *, track_memory_usage: bool = False):
        expected = self.architecture.fused_dim
        if bags.ndim != 3 or bags.shape[-1] != expected:
            raise ValueError(f"paper COMMAND input must have shape (batch, time, {expected}), got {tuple(bags.shape)}")
        appearance = bags[..., : self.architecture.appearance_dim]
        motion = bags[..., self.architecture.appearance_dim :]
        if self.mode == "dual-memory":
            return self.network(appearance, motion, track_memory_usage=track_memory_usage)
        return self.network(appearance, motion)


@dataclass(frozen=True)
class PaperCommandLossBreakdown:
    total: Tensor
    mil: Tensor
    contrastive: Tensor
    focal: Tensor
    anomaly_separation: Tensor
    sparsity: Tensor
    smoothness: Tensor
    score_l2: Tensor

    def detached(self) -> Dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.__dict__.items()}


def paper_command_composite_loss(
    output: PaperCommandOutput,
    weak_labels: Tensor,
    config: PaperCommandLossConfig | None = None,
) -> PaperCommandLossBreakdown:
    """Compute COMMAND's MIL, contrastive, focal, and separation objective."""

    config = config or PaperCommandLossConfig()
    labels = weak_labels.to(dtype=output.logits.dtype).reshape(-1)
    if output.logits.ndim != 2 or len(labels) != output.logits.shape[0]:
        raise ValueError("weak_labels must provide one value per video bag")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("weak_labels must contain only zero and one")

    scores = output.memory.dual_memory_deviation
    bag_scores = scores.amax(dim=1)
    positive = labels > 0.5
    normal = ~positive
    zero = scores.sum() * 0.0

    if positive.any() and normal.any():
        ranking = F.relu(config.mil_margin - bag_scores[positive, None] + bag_scores[None, normal]).mean()
        separation = F.relu(
            config.anomaly_margin - scores[positive].mean(dim=1, keepdim=True) + scores[normal].mean(dim=1).unsqueeze(0)
        ).mean()
    else:
        ranking = zero
        separation = zero

    sparsity = scores[positive].mean() if positive.any() else zero
    smoothness = (scores[:, 1:] - scores[:, :-1]).square().mean() if scores.shape[1] > 1 else zero
    mil = ranking + config.sparsity_weight * sparsity + config.smoothness_weight * smoothness

    projections = F.normalize(output.projections.mean(dim=1), dim=-1)
    contrastive = (
        _supervised_info_nce(projections, labels, config.contrastive_temperature)
        if positive.any() and normal.any()
        else zero
    )

    probabilities = output.probabilities.amax(dim=1).clamp(1e-7, 1.0 - 1e-7)
    positive_probability = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
    alpha = torch.where(labels > 0.5, config.focal_alpha, 1.0 - config.focal_alpha)
    focal = (-alpha * (1.0 - positive_probability).pow(config.focal_gamma) * positive_probability.log()).mean()
    score_l2 = scores.square().mean()

    total = (
        config.mil_weight * mil
        + config.contrastive_weight * contrastive
        + config.focal_weight * focal
        + config.anomaly_separation_weight * separation
        + config.score_l2_weight * score_l2
    )
    return PaperCommandLossBreakdown(
        total=total,
        mil=mil,
        contrastive=contrastive,
        focal=focal,
        anomaly_separation=separation,
        sparsity=sparsity,
        smoothness=smoothness,
        score_l2=score_l2,
    )


def _supervised_info_nce(projections: Tensor, labels: Tensor, temperature: float) -> Tensor:
    if len(projections) < 2:
        return projections.sum() * 0.0
    logits = projections @ projections.transpose(0, 1) / temperature
    identity = torch.eye(len(projections), dtype=torch.bool, device=projections.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return projections.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_positive_log_prob[valid].mean()


def bags_from_concept(
    concept: VideoConcept,
    *,
    task_id: str | None = None,
    expected_windows: int = 32,
) -> Tuple[PaperVideoBag, ...]:
    """Recover complete bags from a video concept without using row strategy matrices."""

    if concept.features.shape[1] != 2048:
        raise ValueError(
            f"paper COMMAND requires 2048-D RGB+flow features; {concept.name!r} has {concept.features.shape[1]}"
        )
    groups: Dict[str, list[int]] = {}
    for index, window in enumerate(concept.windows):
        group_id = str(window.payload.get("record_instance_id", window.video_id))
        groups.setdefault(group_id, []).append(index)
    weak_targets = concept.strategy_targets.get("weak_label")
    bags = []
    for group_id in sorted(groups):
        indices = sorted(
            groups[group_id],
            key=lambda index: (
                int(concept.windows[index].payload.get("window_index", index)),
                concept.windows[index].start_frame,
            ),
        )
        if len(indices) != expected_windows:
            raise ValueError(f"video bag {group_id!r} has {len(indices)} windows; expected {expected_windows}")
        # Test matrices intentionally leave reserved weak-label columns empty.
        # Resolve those bags from metadata, without adding labels to features.
        targets = None if weak_targets is None else np.asarray(weak_targets, dtype=np.float32)[indices]
        if targets is not None and not np.isnan(targets).all():
            labels = targets
        elif all("weak_label" in concept.windows[index].payload for index in indices):
            labels = np.asarray(
                [concept.windows[index].payload["weak_label"] for index in indices],
                dtype=np.float32,
            )
        else:
            labels = np.asarray([concept.windows[index].label for index in indices], dtype=np.float32)
            labels = np.full_like(labels, np.nanmax(labels))
        if not np.isfinite(labels).all() or len(np.unique(labels)) != 1:
            raise ValueError(f"video bag {group_id!r} must resolve to one finite weak label")
        bags.append(
            PaperVideoBag(
                bag_id=group_id,
                task_id=task_id or concept.name,
                features=concept.features[indices],
                weak_label=int(labels[0]),
                windows=tuple(concept.windows[index] for index in indices),
            )
        )
    return tuple(bags)


class ContTrainPlusPlusTrainer:
    """Dedicated complete-bag trainer implementing the ContTrain++ protocol."""

    def __init__(
        self,
        model: PaperCommandVideoModel | None = None,
        config: PaperCommandTrainerConfig | None = None,
    ):
        self.config = config or PaperCommandTrainerConfig()
        self.model = model or PaperCommandVideoModel(self.config.architecture)
        if self.model.architecture != self.config.architecture:
            raise ValueError("model architecture and trainer configuration must match")
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.replay = ContTrainReplayBuffer(self.config.buffer_size)
        self._rng = np.random.default_rng(self.config.seed)
        self._global_epoch = 0
        self._normal_score_history: list[np.ndarray] = []
        self._calibration_median = 0.0
        self._calibration_scale = 1.0
        self.optimizer = self._optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )

    def _optimizer(self):
        secondary = []
        primary = []
        for name, parameter in self.model.named_parameters():
            if "secondary_memory" in name:
                secondary.append(parameter)
            else:
                primary.append(parameter)
        groups = [{"params": primary, "lr": self.config.learning_rate, "name": "primary"}]
        if secondary:
            groups.append(
                {
                    "params": secondary,
                    "lr": self.config.secondary_memory_learning_rate,
                    "name": "secondary_memory",
                }
            )
        return torch.optim.Adam(groups)

    def fit_task(self, bags: Sequence[PaperVideoBag], *, task_id: str | None = None) -> Dict[str, object]:
        if not bags:
            raise ValueError("fit_task requires at least one video bag")
        # ``epochs`` is a per-task setting.  StepLR therefore has to restart at
        # each task boundary; otherwise a 100-epoch first task leaves later
        # tasks at 1e-24 or lower and they cannot adapt at all.  Adam moments
        # and model/replay state remain continual.
        if self._global_epoch:
            self._reset_task_scheduler()
        prepared = tuple(
            PaperVideoBag(
                bag_id=bag.bag_id,
                task_id=task_id or bag.task_id,
                features=bag.features,
                weak_label=bag.weak_label,
                windows=bag.windows,
            )
            for bag in bags
        )
        temporal_lengths = {len(bag.features) for bag in prepared}
        if len(temporal_lengths) != 1:
            raise ValueError("all video bags in a task must have the same temporal length")
        feature_dims = {bag.features.shape[1] for bag in prepared}
        if feature_dims != {self.config.architecture.fused_dim}:
            raise ValueError(
                f"task bags must have feature dimension {self.config.architecture.fused_dim}, got {sorted(feature_dims)}"
            )
        started = time.perf_counter()
        totals: Dict[str, float] = {}
        batches = 0
        novelty_replacements = 0
        latest_scores: Dict[str, np.ndarray] = {}
        self.model.train()
        for _ in range(self.config.epochs):
            permutation = self._rng.permutation(len(prepared))
            current_batch_size = (
                self.config.batch_size - self.config.replay_batch_size if len(self.replay) else self.config.batch_size
            )
            for offset in range(0, len(prepared), current_batch_size):
                current = tuple(prepared[index] for index in permutation[offset : offset + current_batch_size])
                replay = self.replay.sample(
                    self.config.replay_batch_size,
                    exclude_bag_ids=(bag.bag_id for bag in current),
                )
                combined = (*current, *replay)
                features = torch.as_tensor(
                    np.stack([bag.features for bag in combined]),
                    dtype=torch.float32,
                    device=self.device,
                )
                labels = torch.as_tensor(
                    [bag.weak_label for bag in combined],
                    dtype=torch.float32,
                    device=self.device,
                )
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(features, track_memory_usage=self.model.mode == "dual-memory")
                if self.model.mode == "dual-memory":
                    loss = paper_command_composite_loss(output, labels, self.config.loss)
                else:
                    loss = self._normal_only_loss(output)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                novelty_replacements += self._replace_novel_normal_features(output, labels)

                curves = self._raw_scores(output).detach().cpu().numpy().astype(np.float32)
                batch_entries = []
                for index, bag in enumerate(current):
                    latest_scores[bag.bag_id] = curves[index]
                    batch_entries.append(ReplayEntry(bag=bag, score_curve=curves[index]))
                self.replay.add(batch_entries)
                detached = loss.detached()
                for name, value in detached.items():
                    totals[name] = totals.get(name, 0.0) + value
                batches += 1
            self.scheduler.step()
            self._global_epoch += 1

        replay_entries = []
        for bag in prepared:
            curve = latest_scores[bag.bag_id]
            replay_entries.append(ReplayEntry(bag=bag, score_curve=curve))
            if bag.weak_label == 0:
                self._normal_score_history.append(curve.copy())
        self.replay.add(replay_entries)
        self._update_calibration()
        return {
            "task_id": task_id or prepared[0].task_id,
            "bags": len(prepared),
            "epochs": self.config.epochs,
            "batches": batches,
            "loss": {name: value / max(1, batches) for name, value in totals.items()},
            "buffer_bags": len(self.replay),
            "novelty_replacements": novelty_replacements,
            "learning_rates": {
                group.get("name", str(index)): group["lr"] for index, group in enumerate(self.optimizer.param_groups)
            },
            "wall_seconds": time.perf_counter() - started,
        }

    def _reset_task_scheduler(self) -> None:
        for group in self.optimizer.param_groups:
            if group.get("name") == "secondary_memory":
                group["lr"] = self.config.secondary_memory_learning_rate
            else:
                group["lr"] = self.config.learning_rate
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )

    def _normal_only_loss(self, output: PaperNormalOnlyOutput) -> PaperCommandLossBreakdown:
        scores = output.nearest_distance
        compactness = scores.mean()
        smoothness = (scores[:, 1:] - scores[:, :-1]).square().mean() if scores.shape[1] > 1 else scores.sum() * 0
        score_l2 = scores.square().mean()
        total = (
            compactness + self.config.loss.smoothness_weight * smoothness + self.config.loss.score_l2_weight * score_l2
        )
        zero = scores.sum() * 0.0
        return PaperCommandLossBreakdown(total, compactness, zero, zero, zero, zero, smoothness, score_l2)

    @staticmethod
    def _raw_scores(output: PaperCommandOutput | PaperNormalOnlyOutput) -> Tensor:
        if isinstance(output, PaperNormalOnlyOutput):
            return output.nearest_distance
        return output.memory.dual_memory_deviation

    def _replace_novel_normal_features(
        self,
        output: PaperCommandOutput | PaperNormalOnlyOutput,
        labels: Tensor,
    ) -> int:
        if self.model.mode != "dual-memory" or self._global_epoch < self.config.novelty_warmup_epochs:
            return 0
        normal = labels < 0.5
        if not normal.any():
            return 0
        scores = output.memory.dual_memory_deviation[normal].detach()
        flattened = scores.reshape(-1)
        median = flattened.median()
        mad = (flattened - median).abs().median()
        threshold = median + self.config.novelty_mad_scale * mad
        candidates = torch.nonzero(scores > threshold, as_tuple=False)
        if not len(candidates):
            return 0
        candidate_scores = scores[candidates[:, 0], candidates[:, 1]]
        order = torch.argsort(candidate_scores, descending=True, stable=True)
        temporal = output.temporal_features[normal].detach()
        memory = self.model.network.memory
        for candidate in candidates[order]:
            memory.replace_least_accessed("primary", temporal[candidate[0], candidate[1]])
        return int(len(candidates))

    def _update_calibration(self) -> None:
        if not self._normal_score_history:
            return
        values = np.concatenate(self._normal_score_history).astype(np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        self._calibration_median = median
        self._calibration_scale = max(1e-8, 1.4826 * mad)

    def predict_bags(self, bags: Sequence[PaperVideoBag], *, batch_size: int = 8) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not bags:
            return {
                "anomaly_scores": np.empty((0, 0), dtype=np.float32),
                "raw_scores": np.empty((0, 0), dtype=np.float32),
                "classifier_scores": np.empty((0, 0), dtype=np.float32),
            }
        self.model.eval()
        raw_blocks = []
        classifier_blocks = []
        with torch.no_grad():
            for offset in range(0, len(bags), batch_size):
                batch = bags[offset : offset + batch_size]
                features = torch.as_tensor(
                    np.stack([bag.features for bag in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                output = self.model(features)
                raw_blocks.append(self._raw_scores(output).cpu().numpy())
                if isinstance(output, PaperCommandOutput):
                    classifier_blocks.append(output.probabilities.cpu().numpy())
        raw = np.concatenate(raw_blocks).astype(np.float32, copy=False)
        # A logistic map collapses very negative normal-only distances to one
        # float32 value. The arctangent map is equally monotonic and bounded,
        # but retains useful rank resolution in those extreme tails.
        z = (raw.astype(np.float64) - self._calibration_median) / self._calibration_scale
        calibrated = 0.5 + np.arctan(z) / np.pi
        classifier = (
            np.concatenate(classifier_blocks).astype(np.float32, copy=False)
            if classifier_blocks
            else np.full_like(raw, np.nan)
        )
        return {"anomaly_scores": calibrated, "raw_scores": raw, "classifier_scores": classifier}

    def state_dict(self) -> Mapping[str, object]:
        return {
            "config": asdict(self.config),
            "mode": self.model.mode,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "replay": self.replay.state_dict(),
            "global_epoch": self._global_epoch,
            "normal_score_history": self._normal_score_history,
            "calibration_median": self._calibration_median,
            "calibration_scale": self._calibration_scale,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state["mode"] != self.model.mode:
            raise ValueError("checkpoint mode does not match trainer model")
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint configuration does not match trainer configuration")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.replay.load_state_dict(state["replay"])
        self._global_epoch = int(state["global_epoch"])
        self._normal_score_history = [np.asarray(values, dtype=np.float32) for values in state["normal_score_history"]]
        self._calibration_median = float(state["calibration_median"])
        self._calibration_scale = float(state["calibration_scale"])
        self._rng.bit_generator.state = state["rng_state"]

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            torch.save(dict(self.state_dict()), temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path).expanduser().resolve(), map_location=self.device, weights_only=False)
        self.load_state_dict(state)

    def metadata(self) -> Dict[str, object]:
        return {
            "mode": self.model.mode,
            "architecture": asdict(self.config.architecture),
            "training": {
                key: value for key, value in asdict(self.config).items() if key not in {"architecture", "loss"}
            },
            "loss": asdict(self.config.loss),
            "buffer_bags": len(self.replay),
            "global_epochs": self._global_epoch,
            "calibration": {"median": self._calibration_median, "mad_scale": self._calibration_scale},
        }
