"""Learning without Forgetting (LwF) strategy for continual learning."""

import copy
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


class LwFStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    """
    Learning without Forgetting strategy using knowledge distillation.

    Prevents catastrophic forgetting by maintaining a frozen copy of the model
    from the previous task and using it to regularize training on new data.
    The new model learns to both reconstruct new data well AND preserve the
    knowledge encoded in the old model's latent representations.

    Parameters
    ----------
    model : Model
        The continual learning model (must support distillation)
    alpha : float, optional
        Weight for distillation loss (default: 0.5)
        Higher values preserve more old knowledge but may limit adaptation
    distill_mode : str, optional
        Type of distillation: 'latent', 'reconstruction', or 'hybrid' (default: 'latent')
    """

    def __init__(self, model: Model, alpha: float = 0.5, distill_mode: str = "latent"):
        self._model = model
        self._old_model: Optional[Model] = None
        self._alpha = alpha
        self._distill_mode = distill_mode
        self._task_count = 0

        if distill_mode not in ["latent", "reconstruction", "hybrid"]:
            raise ValueError(
                f"Invalid distill_mode: {distill_mode}. " f"Must be 'latent', 'reconstruction', or 'hybrid'"
            )

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        """
        Learn from new data using knowledge distillation if not the first task.

        Parameters
        ----------
        data : np.ndarray
            Training data for the current task/window
        """
        # First task: train normally without distillation
        if self._old_model is None:
            logger.info(f"Task {self._task_count + 1}: Training without distillation (first task)")
            self._model.fit(data)
        else:
            # Subsequent tasks: train with distillation from old model
            logger.info(
                f"Task {self._task_count + 1}: Training with LwF distillation "
                f"(alpha={self._alpha}, mode={self._distill_mode})"
            )
            self._fit_with_distillation(data)

        self._task_count += 1

        # Clone current model as old model for next task
        self._old_model = self._clone_model()

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple:
        """
        Predict anomalies using the current model.

        Parameters
        ----------
        data : np.ndarray
            Data to predict on

        Returns
        -------
        tuple
            (predicted labels, anomaly scores)
        """
        return self._model.predict(data)

    def name(self) -> str:
        """Return strategy name."""
        return "LwF"

    def additional_info(self) -> Dict:
        """Return additional strategy information."""
        return {
            "model": self._model.name(),
            "alpha": self._alpha,
            "distill_mode": self._distill_mode,
            "task_count": self._task_count,
            "has_old_model": self._old_model is not None,
        }

    def _resolve_trainable_module(self, model: Optional[Model] = None) -> Optional[torch.nn.Module]:
        """
        Resolve the underlying trainable torch module from nested model wrappers.
        """
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
        """
        Resolve an attribute from the wrapper chain or the underlying module.
        """
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

    def _transform_data_for_model(self, data: np.ndarray, model: Optional[Model] = None) -> np.ndarray:
        """
        Apply simple wrapper-specific preprocessing so distillation sees the same input shape as fit().
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

    def _supports_latent_distillation(self, model: Optional[Model] = None) -> bool:
        module = self._resolve_trainable_module(model)
        if module is None:
            return False

        return hasattr(module, "encode") or hasattr(module, "encoder")

    @staticmethod
    def _extract_latent_representation(encoded_output):
        if isinstance(encoded_output, (tuple, list)):
            for item in encoded_output:
                if torch.is_tensor(item):
                    return item
        return encoded_output

    def _encode_with_model(self, model: Model, batch: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "encode_batch"):
            return self._extract_latent_representation(model.encode_batch(batch))

        module = self._resolve_trainable_module(model)
        if module is None:
            raise TypeError("Model does not expose a trainable torch module or encode_batch().")

        if hasattr(module, "encode"):
            return self._extract_latent_representation(module.encode(batch))

        if hasattr(module, "encoder"):
            return self._extract_latent_representation(module.encoder(batch))

        raise TypeError("Model does not expose encoder/encode hooks required for latent distillation.")

    def _forward_with_model(
        self, model: Model, batch: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch, apply_masking=False)

        module = self._resolve_trainable_module(model)
        if module is None:
            raise TypeError("Model does not expose a trainable torch module or forward_batch().")

        output = module(batch)
        x_hat = output[0] if isinstance(output, (tuple, list)) else output
        z = self._encode_with_model(model, batch) if self._supports_latent_distillation(model) else None
        return x_hat, z, None

    def _training_batch_step(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None]:
        if hasattr(self._model, "training_batch_step"):
            return self._model.training_batch_step(batch)

        x_hat, z, _ = self._forward_with_model(self._model, batch)
        rec_loss = torch.nn.functional.mse_loss(x_hat, batch)
        return rec_loss, x_hat, z, None

    def _resolve_distill_mode(self) -> str:
        if self._distill_mode in ["latent", "hybrid"] and not (
            self._supports_latent_distillation(self._model) and self._supports_latent_distillation(self._old_model)
        ):
            logger.warning(
                "Latent distillation requested, but the model does not expose encoder hooks. "
                "Falling back to reconstruction distillation."
            )
            return "reconstruction"

        return self._distill_mode

    def _clone_model(self) -> Model:
        """
        Create a frozen deep copy of the current model.

        Returns
        -------
        Model
            Frozen copy of the current model
        """
        cloned = copy.deepcopy(self._model)

        module = self._resolve_trainable_module(cloned)
        if module is not None:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        return cloned

    def _fit_with_distillation(self, data: np.ndarray) -> None:
        """
        Train the model with knowledge distillation from the old model.

        Parameters
        ----------
        data : np.ndarray
            Training data of shape (n_samples, height, width)
        """
        module = self._resolve_trainable_module(self._model)
        old_module = self._resolve_trainable_module(self._old_model)
        if module is None or old_module is None:
            logger.warning(
                "Model does not expose a trainable torch module required for distillation. "
                "Falling back to standard fit()."
            )
            self._model.fit(data)
            return

        tensor_data = self._prepare_data(data)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self._resolve_batch_size(), shuffle=True, num_workers=0)

        distill_mode = self._resolve_distill_mode()
        device = self._resolve_device(self._model)

        old_module.eval()
        for param in old_module.parameters():
            param.requires_grad = False

        old_module.to(device)
        module.to(device)
        module.train()

        optimizer = torch.optim.Adam(module.parameters(), lr=self._resolve_lr())

        for epoch in range(self._resolve_epochs()):
            epoch_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_distill_loss = 0.0

            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(device)

                rec_loss, x_hat_train, z_train, _ = self._training_batch_step(batch)
                x_hat_new, z_new = x_hat_train, z_train

                with torch.no_grad():
                    if distill_mode == "latent":
                        z_old = self._encode_with_model(self._old_model, batch)
                        distill_loss = torch.nn.functional.mse_loss(z_new, z_old)
                    elif distill_mode == "reconstruction":
                        x_hat_old, _, _ = self._forward_with_model(self._old_model, batch)
                        distill_loss = torch.nn.functional.mse_loss(x_hat_new, x_hat_old)
                    elif distill_mode == "hybrid":
                        z_old = self._encode_with_model(self._old_model, batch)
                        x_hat_old, _, _ = self._forward_with_model(self._old_model, batch)
                        latent_loss = torch.nn.functional.mse_loss(z_new, z_old)
                        recon_loss = torch.nn.functional.mse_loss(x_hat_new, x_hat_old)
                        distill_loss = (latent_loss + recon_loss) / 2
                    else:
                        raise ValueError(f"Invalid distill_mode: {distill_mode}")

                total_loss = rec_loss + self._alpha * distill_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_rec_loss += rec_loss.item()
                epoch_distill_loss += distill_loss.item()

            n_batches = len(dataloader)
            avg_loss = epoch_loss / n_batches
            avg_rec = epoch_rec_loss / n_batches
            avg_distill = epoch_distill_loss / n_batches

            epochs = self._resolve_epochs()
            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: " f"Loss={avg_loss:.6f} (Rec={avg_rec:.6f}, Distill={avg_distill:.6f})"
                )

        if hasattr(self._model, "_auto_threshold") and self._model._auto_threshold:
            self._model._calibrate_threshold(data)

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        """
        Prepare numpy data for PyTorch training.

        Parameters
        ----------
        data : np.ndarray
            Training data of shape (n_samples, height, width)

        Returns
        -------
        torch.Tensor
            Prepared tensor data
        """
        transformed = self._transform_data_for_model(data)
        return torch.tensor(transformed, dtype=torch.float32)
