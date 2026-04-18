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
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.strategies.replay.buffers.balanced import BalancedReplayBuffer
from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.replay.replay import ReplayOnlyStrategy

logger = logging.getLogger(__name__)

BatchLoss = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
DataTransform = Callable[[np.ndarray], np.ndarray]


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

    The training contract is explicit: callers pass the trainable module, loss
    function, input transform, and optimizer/data-loader settings directly to
    the strategy instead of relying on hidden model introspection.
    """

    def __init__(
        self,
        model: Model,
        buffer: Optional[ReplayBuffer] = None,
        *,
        module: torch.nn.Module,
        loss_fn: Optional[BatchLoss] = None,
        data_transform: Optional[DataTransform] = None,
        batch_size: int = 32,
        lr: float = 1e-2,
        optimizer: str = "sgd",
        epochs: int = 20,
        device: Union[torch.device, str] = "cpu",
        shuffle: bool = True,
        buffer_size: int = 500,
        seed: int = 7,
    ):
        if not isinstance(module, torch.nn.Module):
            raise TypeError("AGEMStrategy requires 'module' to be a torch.nn.Module.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        optimizer_name = optimizer.lower()
        if optimizer_name not in {"sgd", "adam"}:
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer}")

        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._task_count = 0
        self._concept_to_index: Dict[str, int] = {}
        self._module = module
        self._loss_fn = self.default_loss_fn if loss_fn is None else loss_fn
        self._data_transform = self.identity_transform if data_transform is None else data_transform
        self._batch_size = int(batch_size)
        self._lr = float(lr)
        self._optimizer_name = optimizer_name
        self._epochs = int(epochs)
        self._device = torch.device(device)
        self._shuffle = bool(shuffle)

        replay_buffer = buffer if buffer is not None else BalancedReplayBuffer(max_size=buffer_size, seed=self._seed)
        super().__init__(model, replay_buffer)

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
                "batch_size": self._batch_size,
                "lr": self._lr,
                "optimizer": self._optimizer_name,
                "epochs": self._epochs,
                "device": str(self._device),
                "shuffle": self._shuffle,
            }
        )
        return info

    def _fit(self, current_data: np.ndarray) -> None:
        tensor_data = self._prepare_data(current_data)
        dataloader = DataLoader(
            TensorDataset(tensor_data),
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=0,
        )

        module = self._module
        device = self._device
        module.to(device)
        module.train()

        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise TypeError("AGEMStrategy requires at least one trainable parameter.")

        if self._optimizer_name == "adam":
            logger.warning("AGEM with Adam is an approximation; SGD preserves the standard A-GEM update guarantee.")
            optimizer = torch.optim.Adam(parameters, lr=self._lr)
        else:
            optimizer = torch.optim.SGD(parameters, lr=self._lr)
        epochs = self._epochs

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
                current_loss = self._loss_fn(module, batch)
                current_loss.backward()
                current_grad = self.capture_gradients(parameters)
                final_grad = current_grad

                replay_loss = torch.tensor(0.0, device=device)
                replay_examples = self._sample_replay_examples(self._batch_size)
                if replay_examples is not None:
                    # Then estimate how older concepts would like the model to
                    # move by taking a replay gradient from memory.
                    optimizer.zero_grad()
                    replay_batch = self._prepare_data(replay_examples).to(device)
                    replay_loss = self._loss_fn(module, replay_batch)
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

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        transformed = np.array(self._data_transform(np.asarray(data)), dtype=np.float32, copy=True)
        return torch.tensor(transformed, dtype=torch.float32)

    @staticmethod
    def _forward_batch(module: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        output = module(batch)
        if isinstance(output, (tuple, list)):
            return output[0], tuple(output[1:])
        return output, tuple()

    @staticmethod
    def identity_transform(data: np.ndarray) -> np.ndarray:
        return np.array(data, copy=True)

    @staticmethod
    def default_loss_fn(module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        x_hat, _ = AGEMStrategy._forward_batch(module, batch)
        if x_hat.shape != batch.shape:
            raise ValueError(
                "AGEMStrategy.default_loss_fn expects the module output to have the same shape as the input batch. "
                "Pass a custom loss_fn for models with a different training objective."
            )
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
