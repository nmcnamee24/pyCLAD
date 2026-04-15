"""A-GEM replay strategy for torch-backed reconstruction models.

This implementation treats each incoming batch/window as the current task and
keeps a small replay buffer with examples from earlier concepts. During
training, it computes two gradients:

1. the gradient of the reconstruction loss on the current data;
2. the gradient of the reconstruction loss on a replay batch.

If the current gradient would increase loss on replay data, A-GEM projects it
onto the half-space that does not interfere with the replay gradient. In
practice, this means the model is still free to adapt to the new concept, but
it avoids updates that directly conflict with what it learned from earlier
concepts. After optimization, the new concept is added to the replay buffer so
future tasks can be constrained against it as well.
"""

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.strategies.replay.buffers.balanced import BalancedReplayBuffer
from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.replay.replay import ReplayOnlyStrategy

logger = logging.getLogger(__name__)


class AGEMStrategy(ReplayOnlyStrategy):
    """A-GEM strategy using replay-gradient projection.

    Narrative view of the strategy:
    - learn the current concept with a normal reconstruction objective;
    - check whether that update would hurt replayed older concepts;
    - if it would, replace the update with its projected, less-forgetting
      version;
    - store the current concept in replay memory for future constraints.

    This makes A-GEM a replay-driven strategy, but unlike vanilla replay it
    does not simply retrain on concatenated old and new data. Instead, replay
    is used to shape the gradient direction so the model remains plastic on the
    current task without taking obviously destructive steps on past concepts.
    """

    def __init__(
        self,
        model: Model,
        buffer: Optional[ReplayBuffer] = None,
        *,
        buffer_size: int = 500,
        seed: int = 7,
    ):
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._task_count = 0
        self._concept_to_index: Dict[str, int] = {}

        replay_buffer = buffer if buffer is not None else BalancedReplayBuffer(max_size=buffer_size, seed=self._seed)
        super().__init__(model, replay_buffer)

        if self._resolve_trainable_module() is None:
            raise TypeError(
                "AGEMStrategy requires a PyTorch-backed model exposed via '.module' "
                "or through a nested '.model' wrapper."
            )

    def learn(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> None:
        del kwargs

        current_data = np.asarray(data, dtype=np.float32)
        if len(current_data) == 0:
            return

        concept_name = concept_id or f"concept_{self._task_count}"
        concept_index = self._concept_index(concept_name)
        replay_data = np.array(self._buffer.data(), copy=True)

        logger.info("AGEM learn on %s with %s samples", concept_name, len(current_data))
        self._fit(current_data)

        if isinstance(self._buffer, BalancedReplayBuffer):
            self._buffer.add(
                examples=current_data,
                concept_indices=np.full(len(current_data), concept_index, dtype=np.int64),
            )
        else:
            self._buffer.update(current_data)

        calibration_data = current_data
        if len(replay_data) > 0:
            calibration_data = np.concatenate([replay_data, current_data], axis=0)

        if hasattr(self._model, "_auto_threshold") and self._model._auto_threshold:
            self._model._calibrate_threshold(calibration_data)

        self._task_count += 1

    def predict(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        del concept_id, kwargs
        return self._model.predict(data)

    def name(self) -> str:
        return "AGEM"

    def additional_info(self) -> Dict:
        info = super().additional_info()
        info.update(
            {
                "model": self._model.name(),
                "seed": self._seed,
                "task_count": self._task_count,
                "concept_count": len(self._concept_to_index),
            }
        )
        return info

    def _fit(self, current_data: np.ndarray) -> None:
        module = self._resolve_trainable_module()
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for AGEM.")

        tensor_data = self._prepare_data(current_data)
        dataloader = DataLoader(
            TensorDataset(tensor_data),
            batch_size=self._resolve_batch_size(),
            shuffle=self._resolve_shuffle(),
            num_workers=0,
        )

        device = self._resolve_device()
        module.to(device)
        module.train()

        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise TypeError("AGEMStrategy requires at least one trainable parameter.")

        optimizer = torch.optim.Adam(parameters, lr=self._resolve_lr())
        epochs = self._resolve_epochs()

        for epoch in range(epochs):
            epoch_current = 0.0
            epoch_replay = 0.0
            projection_count = 0

            for (batch,) in dataloader:
                batch = batch.to(device)

                # First compute the gradient for the current concept. This is
                # the update we would normally apply if no replay constraint
                # existed.
                optimizer.zero_grad()
                current_loss = self._compute_batch_loss(module, batch)
                current_loss.backward()
                current_grad = self.capture_gradients(parameters)
                final_grad = current_grad

                replay_loss = torch.tensor(0.0, device=device)
                replay_examples = self._sample_replay_examples(self._resolve_batch_size())
                if replay_examples is not None:
                    # Then estimate how older concepts would like the model to
                    # move by taking a replay gradient from memory.
                    optimizer.zero_grad()
                    replay_batch = self._prepare_data(replay_examples).to(device)
                    replay_loss = self._compute_batch_loss(module, replay_batch)
                    replay_loss.backward()
                    replay_grad = self.capture_gradients(parameters)

                    # A negative dot product means the current update would
                    # interfere with replay performance. In that case A-GEM
                    # projects the current gradient to remove the conflicting
                    # component while preserving as much useful progress as
                    # possible.
                    if torch.dot(current_grad, replay_grad).item() < 0:
                        final_grad = self.project_gradient(current_grad, replay_grad)
                        projection_count += 1

                optimizer.zero_grad()
                self.restore_gradients(parameters, final_grad)
                optimizer.step()

                epoch_current += current_loss.item()
                epoch_replay += replay_loss.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                n_batches = len(dataloader)
                projection_rate = 100.0 * projection_count / n_batches
                logger.info(
                    "AGEM epoch %s/%s: current=%.6f replay=%.6f projections=%s/%s (%.1f%%)",
                    epoch + 1,
                    epochs,
                    epoch_current / n_batches,
                    epoch_replay / n_batches,
                    projection_count,
                    n_batches,
                    projection_rate,
                )

    def _sample_replay_examples(self, batch_size: int) -> Optional[np.ndarray]:
        if isinstance(self._buffer, BalancedReplayBuffer):
            if self._buffer.is_empty():
                return None
            return self._buffer.sample(batch_size)["examples"]

        replay_data = self._buffer.data()
        if len(replay_data) == 0:
            return None

        sample_size = min(int(batch_size), len(replay_data))
        indices = self._rng.choice(len(replay_data), size=sample_size, replace=False)
        return np.asarray(replay_data)[indices]

    def _concept_index(self, concept_id: str) -> int:
        if concept_id not in self._concept_to_index:
            self._concept_to_index[concept_id] = len(self._concept_to_index)
        return self._concept_to_index[concept_id]

    def _resolve_trainable_module(self, model: Optional[Model] = None) -> Optional[torch.nn.Module]:
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

    @staticmethod
    def capture_gradients(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        flat_parts = []
        for parameter in parameters:
            if parameter.grad is None:
                flat_parts.append(torch.zeros(parameter.numel(), device=parameter.device, dtype=parameter.dtype))
            else:
                flat_parts.append(parameter.grad.detach().reshape(-1).clone())

        if not flat_parts:
            return torch.empty(0)

        return torch.cat(flat_parts)

    @staticmethod
    def restore_gradients(parameters: Sequence[torch.nn.Parameter], flat_gradient: torch.Tensor) -> None:
        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            grad_slice = flat_gradient[offset : offset + numel].view_as(parameter)
            parameter.grad = grad_slice.clone().to(device=parameter.device, dtype=parameter.dtype)
            offset += numel

    @staticmethod
    def project_gradient(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if current.numel() == 0 or reference.numel() == 0:
            return current

        reference_norm = torch.dot(reference, reference)
        if reference_norm.item() <= 0:
            return current

        correction = torch.dot(current, reference) / reference_norm
        return current - correction * reference
