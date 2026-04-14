"""Elastic Weight Consolidation (EWC) strategy for continual learning."""

import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptIncrementalStrategy,
)

logger = logging.getLogger(__name__)


class EWCStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    """
    Elastic Weight Consolidation strategy using a diagonal Fisher approximation.

    The strategy is self-contained: it resolves the underlying torch module from
    the pyCLAD model wrapper, runs the optimization loop locally, and estimates
    parameter importance from reconstruction-loss gradients.

    Parameters
    ----------
    model : Model
        Continual learning model backed by a torch module.
    ewc_lambda : float, optional
        Regularization strength (default: 1000.0).
    mode : str, optional
        'separate' to keep one Fisher estimate per task (default).
        'online' is reserved for a future consolidated implementation.
    decay_factor : float, optional
        Reserved for online mode.
    keep_importance_data : bool, optional
        When False in separate mode, only the latest task statistics are kept.
    """

    def __init__(
        self,
        model: Model,
        ewc_lambda: float = 1000.0,
        mode: str = "separate",
        decay_factor: float = 0.9,
        keep_importance_data: bool = True,
    ):
        self._model = model
        self._ewc_lambda = ewc_lambda
        self._mode = mode
        self._decay_factor = decay_factor
        self._keep_importance_data = keep_importance_data

        self._saved_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self._importances: Dict[int, Dict[str, torch.Tensor]] = {}
        self._task_count = 0

        if mode not in ["separate", "online"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'separate' or 'online'")

        if mode == "online":
            raise NotImplementedError("Online mode not yet implemented. Use 'separate' mode.")

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        """
        Learn from new data and update EWC statistics afterwards.

        Parameters
        ----------
        data : np.ndarray
            Training data for the current task/window.
        """
        if self._resolve_trainable_module() is None:
            raise TypeError(
                "EWCStrategy requires a PyTorch-backed model exposed via '.module' "
                "or through a nested '.model' wrapper."
            )

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

        if not self._keep_importance_data:
            self._keep_latest_task_only()

        self._task_count += 1

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple:
        """
        Predict anomalies using the current model.

        Parameters
        ----------
        data : np.ndarray
            Data to predict on.

        Returns
        -------
        tuple
            (predicted labels, anomaly scores)
        """
        return self._model.predict(data)

    def name(self) -> str:
        """Return strategy name."""
        return "EWC"

    def additional_info(self) -> Dict:
        """Return additional strategy information."""
        total_params = sum(sum(p.numel() for p in task_params.values()) for task_params in self._saved_params.values())
        total_importances = sum(
            sum(imp.numel() for imp in task_imps.values()) for task_imps in self._importances.values()
        )

        return {
            "model": self._model.name(),
            "ewc_lambda": self._ewc_lambda,
            "mode": self._mode,
            "task_count": self._task_count,
            "num_saved_tasks": len(self._saved_params),
            "total_stored_params": total_params,
            "total_stored_importances": total_importances,
            "memory_efficient": self._mode == "online",
        }

    def _resolve_trainable_module(self, model: Optional[Model] = None) -> Optional[torch.nn.Module]:
        """Resolve the underlying trainable torch module from nested model wrappers."""
        current = self._model if model is None else model
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            module = getattr(current, "module", None)
            if isinstance(module, torch.nn.Module):
                return module

            module = getattr(current, "model", None)
            if isinstance(module, torch.nn.Module):
                return module

            current = getattr(current, "model", None)

        return None

    def _resolve_model_attr(self, attr_name: str, default, model: Optional[Model] = None):
        """Resolve an attribute from the wrapper chain or the underlying module."""
        current = self._model if model is None else model
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            if hasattr(current, attr_name):
                return getattr(current, attr_name)

            module = getattr(current, "module", None)
            if module is not None and hasattr(module, attr_name):
                return getattr(module, attr_name)

            current = getattr(current, "model", None)

        return default

    def _resolve_batch_size(self) -> int:
        batch_size = self._resolve_model_attr("batch_size", 32)
        return int(batch_size) if batch_size else 32

    def _resolve_lr(self) -> float:
        lr = self._resolve_model_attr("lr", 1e-2)
        return float(lr) if lr else 1e-2

    def _resolve_epochs(self) -> int:
        epochs = self._resolve_model_attr("epochs", 20)
        return int(epochs) if epochs else 20

    def _resolve_device(self, model: Optional[Model] = None) -> torch.device:
        explicit_device = self._resolve_model_attr("device", None, model)
        if explicit_device is not None:
            return torch.device(explicit_device)

        module = self._resolve_trainable_module(model)
        if module is not None:
            try:
                return next(module.parameters()).device
            except (AttributeError, StopIteration):
                pass

        return torch.device("cpu")

    def _resolve_shuffle(self) -> bool:
        current = self._model
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            current_name = current.__class__.__name__
            if current_name in {"TemporalAutoencoder", "VariationalTemporalAutoencoder"}:
                return False
            if current_name == "Autoencoder":
                return True

            module = getattr(current, "module", None)
            if module is not None:
                module_name = module.__class__.__name__
                if module_name in {"TemporalAutoencoderModule", "VariationalTemporalAutoencoderModule"}:
                    return False
                if module_name == "AutoencoderModule":
                    return True

            current = getattr(current, "model", None)

        return True

    def _transform_data_for_model(self, data: np.ndarray, model: Optional[Model] = None) -> np.ndarray:
        """
        Apply lightweight wrapper-specific preprocessing so EWC sees the same
        input shape as the wrapped model.
        """
        current = self._model if model is None else model
        transformed = np.asarray(data)
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            if current.__class__.__name__ == "FlattenTimeSeriesAdapter":
                transformed = transformed.reshape(transformed.shape[0], -1)

            current = getattr(current, "model", None)

        return np.array(transformed, copy=True)

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        transformed = self._transform_data_for_model(data)
        return torch.tensor(transformed, dtype=torch.float32)

    @staticmethod
    def _forward_batch(module: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        output = module(batch)
        if isinstance(output, (tuple, list)):
            return output[0], tuple(output[1:])
        return output, tuple()

    def _compute_batch_loss(self, module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        x_hat, extras = self._forward_batch(module, batch)
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

    def _compute_ewc_penalty(self, module: torch.nn.Module) -> torch.Tensor:
        try:
            penalty = next(module.parameters()).new_tensor(0.0)
        except StopIteration:
            return torch.tensor(0.0)

        if not self._saved_params or not self._importances:
            return penalty

        for task_id, task_params in self._saved_params.items():
            task_importances = self._importances.get(task_id, {})

            for name, param in module.named_parameters():
                if not param.requires_grad or name not in task_params or name not in task_importances:
                    continue

                reference = task_params[name].to(device=param.device, dtype=param.dtype)
                importance = task_importances[name].to(device=param.device, dtype=param.dtype)
                penalty = penalty + (importance * (param - reference).pow(2)).sum()

        return 0.5 * penalty

    def _fit_with_ewc(self, data: np.ndarray) -> None:
        module = self._resolve_trainable_module()
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for EWC.")

        tensor_data = self._prepare_data(data)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self._resolve_batch_size(),
            shuffle=self._resolve_shuffle(),
            num_workers=0,
        )

        device = self._resolve_device()
        module.to(device)
        module.train()

        optimizer = torch.optim.Adam(module.parameters(), lr=self._resolve_lr())
        epochs = self._resolve_epochs()

        for epoch in range(epochs):
            epoch_total_loss = 0.0
            epoch_task_loss = 0.0
            epoch_penalty = 0.0

            for (batch,) in dataloader:
                batch = batch.to(device)

                optimizer.zero_grad()
                task_loss = self._compute_batch_loss(module, batch)
                penalty = self._compute_ewc_penalty(module)
                total_loss = task_loss + self._ewc_lambda * penalty
                total_loss.backward()
                optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_penalty += penalty.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                n_batches = len(dataloader)
                logger.info(
                    "Epoch %s/%s: loss=%.6f (task=%.6f, ewc_penalty=%.6f)",
                    epoch + 1,
                    epochs,
                    epoch_total_loss / n_batches,
                    epoch_task_loss / n_batches,
                    epoch_penalty / n_batches,
                )

        if hasattr(self._model, "_auto_threshold") and self._model._auto_threshold:
            self._model._calibrate_threshold(data)

    def _compute_fisher_information(self, data: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Estimate the diagonal Fisher Information by averaging squared gradients
        of the task loss over the current task data.
        """
        module = self._resolve_trainable_module()
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for Fisher estimation.")

        tensor_data = self._prepare_data(data)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self._resolve_batch_size(), shuffle=False, num_workers=0)

        device = self._resolve_device()
        module.to(device)

        was_training = module.training
        module.eval()

        fisher = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in module.named_parameters()
            if param.requires_grad
        }

        total_samples = 0

        for (batch,) in dataloader:
            batch = batch.to(device)
            batch_size = batch.shape[0]
            total_samples += batch_size

            module.zero_grad()
            loss = self._compute_batch_loss(module, batch)
            loss.backward()

            for name, param in module.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                fisher[name] += param.grad.detach().cpu().pow(2) * batch_size

        if total_samples > 0:
            for name in fisher:
                fisher[name] /= total_samples

        module.zero_grad()
        if was_training:
            module.train()

        return {name: value.detach().clone() for name, value in fisher.items()}

    def _get_current_params(self) -> Dict[str, torch.Tensor]:
        """Snapshot the current trainable parameters."""
        module = self._resolve_trainable_module()
        if module is None:
            return {}

        return {name: param.detach().cpu().clone() for name, param in module.named_parameters() if param.requires_grad}

    def _keep_latest_task_only(self) -> None:
        if not self._saved_params or not self._importances:
            return

        latest_task = max(self._saved_params)
        self._saved_params = {latest_task: self._saved_params[latest_task]}
        self._importances = {latest_task: self._importances[latest_task]}
