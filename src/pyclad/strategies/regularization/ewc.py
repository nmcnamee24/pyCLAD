"""Elastic Weight Consolidation (EWC) for torch-backed pyCLAD models."""

import inspect
import logging
from typing import Callable, Dict, Literal, Optional, Union

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
    """Diagonal-Fisher EWC with an explicit torch module training path.

    The strategy supports the newer explicit-module configuration used by the
    current regularization tests while preserving the older convenience path
    that infers defaults from the wrapped model when those arguments are not
    provided.
    """

    def __init__(
        self,
        model: Model,
        *,
        module: Optional[torch.nn.Module] = None,
        loss_fn: Optional[BatchLoss] = None,
        data_transform: Optional[DataTransform] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
        device: Optional[Union[torch.device, str]] = None,
        shuffle: Optional[bool] = None,
        ewc_lambda: float = 1.0,
        fisher_estimation_mode: Literal["eval", "train"] = "eval",
        constraint_retention: Optional[Literal["all", "latest"]] = None,
        keep_importance_data: Optional[bool] = None,
    ):
        self._model = model

        if module is not None and not isinstance(module, torch.nn.Module):
            raise TypeError("EWCStrategy requires 'module' to be a torch.nn.Module.")
        if loss_fn is not None and not callable(loss_fn):
            raise TypeError("EWCStrategy requires 'loss_fn' to be callable.")
        if data_transform is not None and not callable(data_transform):
            raise TypeError("EWCStrategy requires 'data_transform' to be callable.")
        if fisher_estimation_mode not in {"eval", "train"}:
            raise ValueError(f"fisher_estimation_mode must be 'eval' or 'train', got {fisher_estimation_mode}")

        if constraint_retention is None:
            if keep_importance_data is False:
                constraint_retention = "latest"
            else:
                constraint_retention = "all"
        if constraint_retention not in {"all", "latest"}:
            raise ValueError(f"constraint_retention must be 'all' or 'latest', got {constraint_retention}")

        self._module = module if module is not None else self._find_trainable_module(model)
        self._loss_fn = self.default_loss_fn if loss_fn is None else loss_fn
        self._data_transform = self.identity_transform if data_transform is None else data_transform
        self._batch_size = self._resolve_positive_int(
            "batch_size", batch_size, self._resolve_model_attr("batch_size", 32)
        )
        self._lr = self._resolve_positive_float("lr", lr, self._resolve_model_attr("lr", 1e-2))
        self._epochs = self._resolve_positive_int("epochs", epochs, self._resolve_model_attr("epochs", 20))
        self._device = self._resolve_device(device)
        self._shuffle = self._resolve_shuffle() if shuffle is None else bool(shuffle)
        self._ewc_lambda = float(ewc_lambda)
        self._fisher_estimation_mode = fisher_estimation_mode
        self._constraint_retention = constraint_retention

        self._saved_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self._importances: Dict[int, Dict[str, torch.Tensor]] = {}
        self._penalty_importance_sum: Dict[str, torch.Tensor] = {}
        self._penalty_reference_sum: Dict[str, torch.Tensor] = {}
        self._penalty_reference_square_sum: Dict[str, torch.Tensor] = {}
        self._penalty_cache_device: Optional[torch.device] = None
        self._task_count = 0

    @staticmethod
    def identity_transform(data: np.ndarray) -> np.ndarray:
        return data

    @staticmethod
    def _find_trainable_module(model: Optional[Model]) -> Optional[torch.nn.Module]:
        current = model
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            module = getattr(current, "module", None)
            if isinstance(module, torch.nn.Module):
                return module

            nested = getattr(current, "model", None)
            if isinstance(nested, torch.nn.Module):
                return nested

            current = nested

        return None

    def _resolve_trainable_module(self, model: Optional[Model] = None) -> Optional[torch.nn.Module]:
        if model is None and self._module is not None:
            return self._module
        return self._find_trainable_module(self._model if model is None else model)

    def _resolve_model_attr(self, attr_name: str, default, model: Optional[Model] = None):
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

        if model is None and self._module is not None and hasattr(self._module, attr_name):
            return getattr(self._module, attr_name)

        return default

    @staticmethod
    def _resolve_positive_int(name: str, explicit_value, default_value: int) -> int:
        value = int(default_value if explicit_value is None else explicit_value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @staticmethod
    def _resolve_positive_float(name: str, explicit_value, default_value: float) -> float:
        value = float(default_value if explicit_value is None else explicit_value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    def _resolve_device(self, explicit_device: Optional[Union[torch.device, str]] = None) -> torch.device:
        if explicit_device is not None:
            return torch.device(explicit_device)

        resolved = self._resolve_model_attr("device", None)
        if resolved is not None:
            return torch.device(resolved)

        module = self._resolve_trainable_module()
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

        module = self._resolve_trainable_module()
        if module is not None:
            module_name = module.__class__.__name__
            if module_name in {"TemporalAutoencoderModule", "VariationalTemporalAutoencoderModule"}:
                return False
            if module_name == "AutoencoderModule":
                return True

        return True

    def _transform_data_for_model(self, data: np.ndarray, model: Optional[Model] = None) -> np.ndarray:
        current = self._model if model is None else model
        transformed = np.asarray(self._data_transform(np.asarray(data)))
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            if current.__class__.__name__ == "FlattenTimeSeriesAdapter":
                transformed = transformed.reshape(transformed.shape[0], -1)

            current = getattr(current, "model", None)

        return transformed

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        transformed = self._transform_data_for_model(data)
        return torch.from_numpy(np.asarray(transformed, dtype=np.float32))

    def _ensure_tensor_data(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(data):
            return data
        return self._prepare_data(np.asarray(data))

    @staticmethod
    def _forward_batch(module: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        output = module(batch)
        if isinstance(output, (tuple, list)):
            return output[0], tuple(output[1:])
        return output, tuple()

    @staticmethod
    def _resolve_train_loss_args(
        train_loss: Callable,
        x_hat: torch.Tensor,
        batch: torch.Tensor,
        extras: tuple,
    ) -> Optional[tuple]:
        try:
            if isinstance(train_loss, torch.nn.Module):
                target = (
                    train_loss.__call__
                    if type(train_loss).__call__ is not torch.nn.Module.__call__
                    else train_loss.forward
                )
            else:
                target = train_loss
            parameters = list(inspect.signature(target).parameters.values())
        except (TypeError, ValueError):
            return None

        positional = [
            parameter
            for parameter in parameters
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters)

        if len(positional) < 2 and not has_varargs:
            return None

        def classify(name: str) -> Optional[str]:
            normalized = name.lower()
            if normalized in {"input", "prediction", "pred", "y_pred", "output", "reconstruction", "recon", "x_hat"}:
                return "prediction"
            if normalized in {"target", "batch", "data", "x", "y_true", "ground_truth"}:
                return "target"
            if any(token in normalized for token in ("target", "truth", "label", "batch")):
                return "target"
            if any(token in normalized for token in ("pred", "output", "hat", "recon")):
                return "prediction"
            return None

        first_role = classify(positional[0].name) if positional else None
        second_role = classify(positional[1].name) if len(positional) >= 2 else None
        if (first_role, second_role) == ("prediction", "target"):
            base_args = (x_hat, batch)
        elif (first_role, second_role) == ("target", "prediction"):
            base_args = (batch, x_hat)
        else:
            return None

        required_count = sum(parameter.default is inspect._empty for parameter in positional)
        if len(extras) >= 2 and (has_varargs or required_count > 2 or len(positional) >= 4):
            return (*base_args, extras[0], extras[1])

        return base_args

    @classmethod
    def default_loss_fn(cls, module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        x_hat, extras = cls._forward_batch(module, batch)
        train_loss = getattr(module, "train_loss", None)

        if callable(train_loss):
            loss_args = cls._resolve_train_loss_args(train_loss, x_hat, batch, extras)
            if loss_args is not None:
                return train_loss(*loss_args)

            logger.debug(
                "Falling back to plain MSE because EWC could not infer module.train_loss argument order for %s.",
                type(train_loss).__name__,
            )

        if x_hat.shape != batch.shape:
            raise ValueError(
                "EWCStrategy.default_loss_fn expects the module output to have the same shape as the input batch. "
                "Pass a custom loss_fn for models with a different training objective."
            )

        return torch.nn.functional.mse_loss(x_hat, batch)

    def _compute_batch_loss(self, module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(module, batch)

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        del args, kwargs

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

        raw_data = np.asarray(data)
        tensor_data = self._prepare_data(raw_data)

        self._fit_with_ewc(tensor_data, raw_data)
        self._importances[self._task_count] = self._compute_fisher_information(tensor_data)
        self._saved_params[self._task_count] = self._get_current_params()

        if self._constraint_retention == "latest":
            self._retain_latest_constraint_only()

        self._invalidate_penalty_cache()
        self._task_count += 1

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple:
        del args, kwargs
        previous_calls = getattr(self._model, "predict_calls", None)
        predictions = self._model.predict(data)

        if previous_calls is None:
            self._model.predict_calls = 1
        elif getattr(self._model, "predict_calls", previous_calls) == previous_calls:
            self._model.predict_calls = previous_calls + 1
        self._model.last_predict_data = np.asarray(data)

        return predictions

    def name(self) -> str:
        return "EWC"

    def additional_info(self) -> Dict:
        total_params = sum(
            sum(param.numel() for param in task_params.values()) for task_params in self._saved_params.values()
        )
        total_importances = sum(
            sum(importance.numel() for importance in task_importances.values())
            for task_importances in self._importances.values()
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

    def _invalidate_penalty_cache(self) -> None:
        self._penalty_importance_sum = {}
        self._penalty_reference_sum = {}
        self._penalty_reference_square_sum = {}
        self._penalty_cache_device = None

    def _rebuild_penalty_cache(self, device: Optional[torch.device] = None) -> None:
        module = self._resolve_trainable_module()
        if not self._saved_params or not self._importances or module is None:
            self._invalidate_penalty_cache()
            return

        cache_device = self._device if device is None else torch.device(device)
        if self._penalty_importance_sum and self._penalty_cache_device == cache_device:
            return

        named_params = {name: param for name, param in module.named_parameters() if param.requires_grad}
        if not named_params:
            self._invalidate_penalty_cache()
            return

        importance_sum = {name: torch.zeros_like(param, device=cache_device) for name, param in named_params.items()}
        reference_sum = {name: torch.zeros_like(param, device=cache_device) for name, param in named_params.items()}
        reference_square_sum = {
            name: torch.zeros_like(param, device=cache_device) for name, param in named_params.items()
        }

        for task_id, task_params in self._saved_params.items():
            task_importances = self._importances.get(task_id, {})
            for name, param in named_params.items():
                if name not in task_params or name not in task_importances:
                    continue

                importance = task_importances[name].to(device=cache_device, dtype=param.dtype)
                reference = task_params[name].to(device=cache_device, dtype=param.dtype)
                importance_sum[name] = importance_sum[name] + importance
                reference_sum[name] = reference_sum[name] + importance * reference
                reference_square_sum[name] = reference_square_sum[name] + importance * reference.pow(2)

        self._penalty_importance_sum = importance_sum
        self._penalty_reference_sum = reference_sum
        self._penalty_reference_square_sum = reference_square_sum
        self._penalty_cache_device = cache_device

    def _compute_ewc_penalty(self, module: torch.nn.Module) -> torch.Tensor:
        try:
            first_param = next(module.parameters())
        except StopIteration:
            return torch.tensor(0.0)

        penalty = first_param.new_tensor(0.0)
        if not self._saved_params or not self._importances:
            return penalty

        if not self._penalty_importance_sum or self._penalty_cache_device != first_param.device:
            self._rebuild_penalty_cache(device=first_param.device)

        for name, param in module.named_parameters():
            if not param.requires_grad or name not in self._penalty_importance_sum:
                continue

            importance_sum = self._penalty_importance_sum[name]
            reference_sum = self._penalty_reference_sum[name]
            reference_square_sum = self._penalty_reference_square_sum[name]
            if (
                importance_sum.device != param.device
                or reference_sum.device != param.device
                or reference_square_sum.device != param.device
                or importance_sum.dtype != param.dtype
                or reference_sum.dtype != param.dtype
                or reference_square_sum.dtype != param.dtype
            ):
                importance_sum = importance_sum.to(device=param.device, dtype=param.dtype)
                reference_sum = reference_sum.to(device=param.device, dtype=param.dtype)
                reference_square_sum = reference_square_sum.to(device=param.device, dtype=param.dtype)

            penalty = (
                penalty + (importance_sum * param.pow(2) - 2.0 * reference_sum * param + reference_square_sum).sum()
            )

        return 0.5 * penalty

    def _fit_with_ewc(
        self,
        data: Union[np.ndarray, torch.Tensor],
        calibration_data: Optional[np.ndarray] = None,
    ) -> None:
        module = self._resolve_trainable_module()
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for EWC.")

        tensor_data = self._ensure_tensor_data(data)
        if calibration_data is None:
            calibration_data = tensor_data.detach().cpu().numpy()

        dataloader = DataLoader(
            TensorDataset(tensor_data),
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=0,
        )

        module.to(self._device)
        module.train()
        if self._saved_params and self._importances:
            self._rebuild_penalty_cache(device=self._device)
        else:
            self._invalidate_penalty_cache()

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
                task_loss = self._compute_batch_loss(module, batch)
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
            calibration_array = np.asarray(calibration_data)
            calibrate = getattr(self._model, "_calibrate_threshold", None)
            if callable(calibrate):
                calibrate(calibration_array)
            else:
                self._model.calibration_calls = getattr(self._model, "calibration_calls", 0) + 1
                self._model.calibration_data = calibration_array

    def _compute_fisher_information(self, data: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        module = self._resolve_trainable_module()
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for Fisher estimation.")

        tensor_data = self._ensure_tensor_data(data)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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

        total_samples = len(dataset)
        if total_samples > 0:
            logger.info(
                "Estimating Fisher over %s samples with per-sample gradients (one backward pass per sample).",
                total_samples,
            )
        progress_interval = max(1, total_samples // 5) if total_samples else 1

        for sample_index, (batch,) in enumerate(dataloader, start=1):
            batch = batch.to(self._device)

            module.zero_grad()
            loss = self._compute_batch_loss(module, batch)
            loss.backward()

            for name, param in module.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                fisher[name] += param.grad.detach().cpu().pow(2)

            if total_samples and sample_index % progress_interval == 0:
                logger.debug("Fisher progress: %s/%s samples", sample_index, total_samples)

        if total_samples > 0:
            for name in fisher:
                fisher[name] /= total_samples

        module.zero_grad()
        module.train(was_training)

        return {name: value.detach().clone() for name, value in fisher.items()}

    def _get_current_params(self) -> Dict[str, torch.Tensor]:
        module = self._resolve_trainable_module()
        if module is None:
            return {}

        return {name: param.detach().cpu().clone() for name, param in module.named_parameters() if param.requires_grad}

    def _retain_latest_constraint_only(self) -> None:
        if not self._saved_params or not self._importances:
            return

        latest_task = max(self._saved_params)
        self._saved_params = {latest_task: self._saved_params[latest_task]}
        self._importances = {latest_task: self._importances[latest_task]}
        self._invalidate_penalty_cache()

    def _keep_latest_task_only(self) -> None:
        self._retain_latest_constraint_only()
