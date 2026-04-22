"""Elastic Weight Consolidation (EWC) strategy for continual learning.

This implementation keeps a snapshot of model parameters after each learned
task and estimates a diagonal Fisher Information Matrix from the current task's
reconstruction loss. During training on later tasks, it adds the standard EWC
quadratic penalty so parameters that were important for previous tasks are
discouraged from drifting too far:

    L_total = L_task + lambda * 0.5 * sum(F_i * (theta_i - theta_i^*)^2)

where ``theta_i^*`` is the parameter value stored after an earlier task and
``F_i`` is the corresponding diagonal Fisher importance estimate.
"""

import logging
from typing import Callable, Dict, Literal, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptIncrementalStrategy,
)

logger = logging.getLogger(__name__)

BatchLoss = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
DataTransform = Callable[[np.ndarray], np.ndarray]


class EWCStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    """Elastic Weight Consolidation strategy using a diagonal Fisher approximation.

    Parameters
    ----------
    model : Model
        Outer pyCLAD model used for prediction and optional threshold calibration.
    module : torch.nn.Module
        Trainable torch module optimized by EWC.
    loss_fn : BatchLoss
        Callable receiving ``(module, batch)`` and returning the scalar task loss.
    data_transform : DataTransform
        Callable that maps raw numpy input into the shape expected by ``module``.
    ewc_lambda : float, optional
        Regularization strength. Defaults to ``1.0`` as a conservative starting
        point for reconstruction-style losses. Tune this value to match the
        scale of the task loss and Fisher magnitudes for your model.
    fisher_estimation_mode : {"eval", "train"}, optional
        Module mode used while estimating Fisher information. ``"eval"`` is the
        default for deterministic, stable Fisher estimates.
    constraint_retention : {"all", "latest"}, optional
        Retention policy for stored EWC constraints. ``"all"`` preserves the
        standard multi-task EWC behavior. ``"latest"`` keeps only the most
        recent task's parameter snapshot and Fisher estimate.

    Notes
    -----
    With ``constraint_retention="all"``, the total penalty grows as more tasks
    are retained. This is the standard separate-EWC trade-off, not online EWC.
    Use ``"latest"`` if you want latest-task-only retention with bounded
    constraint growth.

    When the wrapped model enables automatic threshold calibration, EWC
    calibrates on the current task data only because it does not retain raw
    examples from prior tasks.
    """

    def __init__(
        self,
        model: Model,
        *,
        module: torch.nn.Module,
        loss_fn: BatchLoss,
        data_transform: DataTransform,
        batch_size: int,
        lr: float,
        epochs: int,
        device: Union[torch.device, str],
        shuffle: bool,
        ewc_lambda: float = 10000.0,
        fisher_estimation_mode: Literal["eval", "train"] = "eval",
        constraint_retention: Literal["all", "latest"] = "all",
    ):
        if not isinstance(module, torch.nn.Module):
            raise TypeError("EWCStrategy requires 'module' to be a torch.nn.Module.")
        if not callable(loss_fn):
            raise TypeError("EWCStrategy requires 'loss_fn' to be callable.")
        if not callable(data_transform):
            raise TypeError("EWCStrategy requires 'data_transform' to be callable.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if fisher_estimation_mode not in {"eval", "train"}:
            raise ValueError(f"fisher_estimation_mode must be 'eval' or 'train', got {fisher_estimation_mode}")
        if constraint_retention not in {"all", "latest"}:
            raise ValueError(f"constraint_retention must be 'all' or 'latest', got {constraint_retention}")

        self._model = model
        self._module = module
        self._loss_fn = loss_fn
        self._data_transform = data_transform
        self._batch_size = int(batch_size)
        self._lr = float(lr)
        self._epochs = int(epochs)
        self._device = torch.device(device)
        self._shuffle = bool(shuffle)
        self._ewc_lambda = float(ewc_lambda)
        self._fisher_estimation_mode = fisher_estimation_mode
        self._constraint_retention = constraint_retention

        self._saved_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self._importances: Dict[int, Dict[str, torch.Tensor]] = {}
        self._penalty_importance_sum: Dict[str, torch.Tensor] = {}
        self._penalty_reference_sum: Dict[str, torch.Tensor] = {}
        self._penalty_reference_square_sum: Dict[str, torch.Tensor] = {}
        self._task_count = 0

    @staticmethod
    def identity_transform(data: np.ndarray) -> np.ndarray:
        return np.array(data, copy=True)

    @staticmethod
    def default_loss_fn(module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        output = module(batch)
        if isinstance(output, (tuple, list)):
            x_hat = output[0]
            extras = tuple(output[1:])
        else:
            x_hat = output
            extras = tuple()

        train_loss = getattr(module, "train_loss", None)
        if callable(train_loss):
            try:
                if len(extras) >= 2:
                    return train_loss(x_hat, batch, extras[0], extras[1])
                return train_loss(x_hat, batch)
            except TypeError:
                try:
                    if len(extras) >= 2:
                        return train_loss(batch, x_hat, extras[0], extras[1])
                    return train_loss(batch, x_hat)
                except TypeError:
                    pass

        return torch.nn.functional.mse_loss(x_hat, batch)

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        """Learn from new data and update EWC statistics afterwards."""
        del args, kwargs

        if self._task_count == 0:
            logger.info("Task %s: training without EWC penalty (first task)", self._task_count + 1)
        else:
            logger.info(
                "Task %s: training with EWC penalty (lambda=%s)",
                self._task_count + 1,
                self._ewc_lambda,
            )

        self._fit_with_ewc(data)

        self._importances[self._task_count] = self._compute_fisher_information(data)
        self._saved_params[self._task_count] = self._get_current_params()

        if self._constraint_retention == "latest":
            self._retain_latest_constraint_only()

        self._rebuild_penalty_cache()
        self._task_count += 1

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple:
        """Predict anomalies using the current model."""
        del args, kwargs
        return self._model.predict(data)

    def name(self) -> str:
        return "EWC"

    def additional_info(self) -> Dict:
        total_params = sum(sum(p.numel() for p in task_params.values()) for task_params in self._saved_params.values())
        total_importances = sum(
            sum(imp.numel() for imp in task_imps.values()) for task_imps in self._importances.values()
        )

        return {
            "model": self._model.name(),
            "ewc_lambda": self._ewc_lambda,
            "batch_size": self._batch_size,
            "lr": self._lr,
            "epochs": self._epochs,
            "device": str(self._device),
            "shuffle": self._shuffle,
            "fisher_estimation_mode": self._fisher_estimation_mode,
            "constraint_retention": self._constraint_retention,
            "task_count": self._task_count,
            "num_saved_tasks": len(self._saved_params),
            "total_stored_params": total_params,
            "total_stored_importances": total_importances,
        }

    def _compute_ewc_penalty(self, module: torch.nn.Module) -> torch.Tensor:
        try:
            penalty = next(module.parameters()).new_tensor(0.0)
        except StopIteration:
            return torch.tensor(0.0)

        if not self._saved_params or not self._importances:
            return penalty

        if not self._penalty_importance_sum:
            self._rebuild_penalty_cache()

        for name, param in module.named_parameters():
            if not param.requires_grad or name not in self._penalty_importance_sum:
                continue

            importance_sum = self._penalty_importance_sum[name].to(device=param.device, dtype=param.dtype)
            reference_sum = self._penalty_reference_sum[name].to(device=param.device, dtype=param.dtype)
            reference_square_sum = self._penalty_reference_square_sum[name].to(device=param.device, dtype=param.dtype)

            penalty = penalty + (importance_sum * param.pow(2) - 2 * reference_sum * param + reference_square_sum).sum()

        return 0.5 * penalty

    def _fit_with_ewc(self, data: np.ndarray) -> None:
        transformed = np.array(self._data_transform(np.asarray(data)), dtype=np.float32, copy=True)
        tensor_data = torch.tensor(transformed, dtype=torch.float32)
        dataloader = DataLoader(
            TensorDataset(tensor_data),
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=0,
        )

        module = self._module
        module.to(self._device)
        module.train()

        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise TypeError("EWCStrategy requires at least one trainable parameter.")

        optimizer = torch.optim.Adam(parameters, lr=self._lr)

        for epoch in range(self._epochs):
            epoch_total_loss = 0.0
            epoch_task_loss = 0.0
            epoch_penalty = 0.0

            for (batch,) in dataloader:
                batch = batch.to(self._device)

                optimizer.zero_grad()
                task_loss = self._loss_fn(module, batch)
                penalty = self._compute_ewc_penalty(module)
                total_loss = task_loss + self._ewc_lambda * penalty
                total_loss.backward()
                optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_penalty += penalty.item()

            if (epoch + 1) % max(1, self._epochs // 5) == 0:
                n_batches = len(dataloader)
                logger.info(
                    "Epoch %s/%s: loss=%.6f (task=%.6f, ewc_penalty=%.6f)",
                    epoch + 1,
                    self._epochs,
                    epoch_total_loss / n_batches,
                    epoch_task_loss / n_batches,
                    epoch_penalty / n_batches,
                )

        if hasattr(self._model, "_auto_threshold") and self._model._auto_threshold:
            # EWC does not retain raw historical examples, so threshold
            # calibration can only use the current task data.
            self._model._calibrate_threshold(data)

    def _compute_fisher_information(self, data: np.ndarray) -> Dict[str, torch.Tensor]:
        """Estimate the diagonal Fisher Information from per-sample gradients."""
        transformed = np.array(self._data_transform(np.asarray(data)), dtype=np.float32, copy=True)
        tensor_data = torch.tensor(transformed, dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(tensor_data), batch_size=1, shuffle=False, num_workers=0)

        module = self._module
        module.to(self._device)

        was_training = module.training
        if self._fisher_estimation_mode == "train":
            module.train()
        else:
            module.eval()

        fisher = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in module.named_parameters()
            if param.requires_grad
        }

        total_samples = 0

        for (batch,) in dataloader:
            batch = batch.to(self._device)
            total_samples += batch.shape[0]

            module.zero_grad()
            loss = self._loss_fn(module, batch)
            loss.backward()

            for name, param in module.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                fisher[name] += param.grad.detach().cpu().pow(2)

        if total_samples > 0:
            for name in fisher:
                fisher[name] /= total_samples

        module.zero_grad()
        module.train(was_training)

        return {name: value.detach().clone() for name, value in fisher.items()}

    def _get_current_params(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().clone() for name, param in self._module.named_parameters() if param.requires_grad
        }

    def _retain_latest_constraint_only(self) -> None:
        if not self._saved_params or not self._importances:
            return

        latest_task = max(self._saved_params)
        self._saved_params = {latest_task: self._saved_params[latest_task]}
        self._importances = {latest_task: self._importances[latest_task]}

    def _rebuild_penalty_cache(self) -> None:
        self._penalty_importance_sum = {}
        self._penalty_reference_sum = {}
        self._penalty_reference_square_sum = {}

        for task_id, task_params in self._saved_params.items():
            task_importances = self._importances.get(task_id, {})

            for name, reference in task_params.items():
                if name not in task_importances:
                    continue

                importance = task_importances[name]
                weighted_reference = importance * reference
                weighted_reference_square = importance * reference.pow(2)

                if name not in self._penalty_importance_sum:
                    self._penalty_importance_sum[name] = importance.detach().clone()
                    self._penalty_reference_sum[name] = weighted_reference.detach().clone()
                    self._penalty_reference_square_sum[name] = weighted_reference_square.detach().clone()
                    continue

                self._penalty_importance_sum[name] += importance
                self._penalty_reference_sum[name] += weighted_reference
                self._penalty_reference_square_sum[name] += weighted_reference_square
