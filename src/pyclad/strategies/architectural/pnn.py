"""Progressive Neural Networks architectural strategy for anomaly detection."""

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import TensorDataset

from pyclad.models.model import Model
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy

AdapterFactory = Callable[[torch.Size, torch.Size], nn.Module]


class _PNNAutoencoderColumn(nn.Module):
    """A single frozen-or-trainable PNN autoencoder column."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_previous_columns: int,
        adapter_factory: Optional[AdapterFactory],
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_previous_columns = num_previous_columns
        self.adapter_factory = adapter_factory
        self.adapters = nn.ModuleList()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _make_adapter(self, old_shape: torch.Size, new_shape: torch.Size, device: torch.device) -> nn.Module:
        if self.adapter_factory is None:
            if old_shape != new_shape:
                raise ValueError(
                    "Identity adapters require matching latent shapes. "
                    f"Received old latent shape {tuple(old_shape)} and new latent shape {tuple(new_shape)}. "
                    "Provide adapter_factory to define a projection."
                )
            return nn.Identity().to(device)

        adapter = self.adapter_factory(old_shape, new_shape)
        if not isinstance(adapter, nn.Module):
            raise TypeError("adapter_factory must return an nn.Module instance.")
        return adapter.to(device)

    def _ensure_adapters(self, old_latents: List[torch.Tensor], z_new: torch.Tensor) -> None:
        if len(old_latents) != self.num_previous_columns:
            raise ValueError(f"Column expects {self.num_previous_columns} old latents but received {len(old_latents)}.")
        if self.adapters:
            return

        self.adapters.extend(
            self._make_adapter(old_latent.shape[1:], z_new.shape[1:], z_new.device) for old_latent in old_latents
        )

    def forward(
        self, x: torch.Tensor, old_latents: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_latents = [] if old_latents is None else list(old_latents)
        if len(old_latents) != self.num_previous_columns:
            raise ValueError(f"Column expects {self.num_previous_columns} old latents but received {len(old_latents)}.")

        z_new = self.encoder(x)
        if not torch.is_tensor(z_new):
            raise TypeError("PNNStrategy expects encoders to return a tensor latent representation.")

        if old_latents:
            self._ensure_adapters(old_latents, z_new)
            lateral = torch.stack(
                [adapter(old_latent) for adapter, old_latent in zip(self.adapters, old_latents)], dim=0
            ).sum(dim=0)
            z_new = z_new + lateral

        return self.decoder(z_new), z_new


class _PNNAutoencoderModule(pl.LightningModule):
    """Internal Lightning module that trains the active PNN autoencoder column."""

    def __init__(self, adapter_factory: Optional[AdapterFactory], freeze_old_columns: bool, lr: float = 1e-3):
        super().__init__()
        self.adapter_factory = adapter_factory
        self.freeze_old_columns = freeze_old_columns
        self.lr = lr
        self.columns = nn.ModuleList()
        self.current_task = -1
        self.loss_fn = nn.MSELoss()

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    def _current_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def add_column(self, encoder: nn.Module, decoder: nn.Module) -> None:
        if self.freeze_old_columns:
            for column in self.columns:
                column.freeze()

        self.columns.append(
            _PNNAutoencoderColumn(
                encoder=encoder,
                decoder=decoder,
                num_previous_columns=self.num_columns,
                adapter_factory=self.adapter_factory,
            ).to(self._current_device())
        )
        self.current_task = self.num_columns - 1

    def _old_latents(self, x: torch.Tensor, task_label: int) -> List[torch.Tensor]:
        latents = []
        for column in self.columns[:task_label]:
            with torch.no_grad():
                _, latent = column(x, latents)
            latents.append(latent.detach())
        return latents

    def forward(self, x: torch.Tensor, task_label: Optional[int] = None) -> torch.Tensor:
        task_label = self.current_task if task_label is None else task_label
        if task_label < 0 or task_label >= self.num_columns:
            raise ValueError(f"Invalid task label {task_label}. Available tasks: 0..{self.num_columns - 1}.")

        x_hat, _ = self.columns[task_label](x, self._old_latents(x, task_label))
        return x_hat

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.loss_fn(self(x), x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, self.parameters()), lr=self.lr)


class PNNStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """
    Progressive Neural Networks architectural strategy for anomaly detection.

    The strategy keeps reconstruction error as the anomaly score while growing
    the underlying autoencoder architecture across continual experiences.
    """

    def __init__(
        self,
        base_model_factory: Callable[[], Model],
        adapter_factory: Optional[AdapterFactory] = None,
        task_free: bool = False,
        freeze_old_columns: bool = True,
        threshold: Optional[float] = None,
        auto_add_column: bool = False,
        random_state: Optional[int] = None,
    ):
        if not callable(base_model_factory):
            raise TypeError("base_model_factory must be callable and return a fresh autoencoder-style model.")

        self.base_model_factory = base_model_factory
        self.task_free = task_free
        self.freeze_old_columns = freeze_old_columns
        self.threshold = threshold
        self.auto_add_column = auto_add_column
        self.random_state = random_state
        self.module = _PNNAutoencoderModule(adapter_factory, freeze_old_columns)
        self._configs: List[Dict[str, float]] = []
        self._trained_columns: Set[int] = set()
        self._concept_to_task: Dict[str, int] = {}
        self._base_model_name: Optional[str] = None

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.end_task()

    @property
    def current_task(self) -> int:
        return self.module.current_task

    @property
    def num_columns(self) -> int:
        return self.module.num_columns

    @staticmethod
    def _extract_column_parts(model: Model) -> Tuple[nn.Module, nn.Module, Dict[str, float], float, str]:
        module = getattr(model, "module", None)
        encoder = getattr(module, "encoder", None)
        decoder = getattr(module, "decoder", None)
        if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
            raise TypeError(
                "base_model_factory must return an autoencoder-style model exposing module.encoder and module.decoder."
            )

        threshold = getattr(model, "threshold", None)
        if threshold is None:
            raise ValueError("base_model_factory must return a model exposing threshold, or threshold must be set.")

        return (
            encoder,
            decoder,
            {
                "lr": float(getattr(module, "lr", getattr(model, "lr", 1e-2))),
                "epochs": int(getattr(model, "epochs", 20) or 20),
                "batch_size": int(getattr(model, "batch_size", 32) or 32),
            },
            float(threshold),
            model.name(),
        )

    def _trainer(self, epochs: int) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    def _current_device(self) -> torch.device:
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _to_numpy(data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _reconstruction_error(data: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        return ((data - x_hat) ** 2).reshape(len(data), -1).mean(axis=1)

    def _scores_for_task(self, data: np.ndarray, task_label: int) -> np.ndarray:
        data = self._to_numpy(data)
        tensor_data = torch.as_tensor(data, dtype=torch.float32, device=self._current_device())
        self.module.eval()
        with torch.no_grad():
            x_hat = self.module(tensor_data, task_label=task_label).detach().cpu().numpy()
        return self._reconstruction_error(data, x_hat)

    def fit(self, data: np.ndarray) -> None:
        data = self._to_numpy(data)
        if len(data) == 0:
            raise ValueError("PNNStrategy.fit received an empty dataset.")

        config = self._configs[self.current_task]
        tensor_data = torch.as_tensor(data, dtype=torch.float32)
        dataloader = torch.utils.data.DataLoader(
            TensorDataset(tensor_data),
            batch_size=config["batch_size"],
            shuffle=True,
        )

        self.module.lr = config["lr"]
        sample = tensor_data[: min(len(tensor_data), config["batch_size"])].to(self._current_device())
        _ = self.module(sample, task_label=self.current_task)

        self._trainer(config["epochs"]).fit(self.module, dataloader)
        self._trained_columns.add(self.current_task)

        if self.auto_add_column:
            self.end_task()

    def learn(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> None:
        if concept_id is None:
            if self.current_task in self._trained_columns:
                self.end_task()
        elif concept_id in self._concept_to_task:
            if self._concept_to_task[concept_id] != self.current_task:
                raise ValueError(
                    f"Concept '{concept_id}' is already assigned to frozen task {self._concept_to_task[concept_id]}."
                )
        else:
            if self.current_task in self._trained_columns:
                self.end_task()
            self._concept_to_task[concept_id] = self.current_task

        self.fit(data)

    def predict(
        self,
        data: np.ndarray,
        task_label: Optional[int] = None,
        concept_id: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.task_free and task_label is None and concept_id in self._concept_to_task:
            task_label = self._concept_to_task[concept_id]

        if task_label is None:
            task_labels = (
                sorted(self._trained_columns) if self.task_free and self._trained_columns else [self.current_task]
            )
            scores = [self._scores_for_task(data, label) for label in task_labels]
            final_scores = scores[0] if len(scores) == 1 else np.stack(scores, axis=0).min(axis=0)
        else:
            final_scores = self._scores_for_task(data, task_label)

        return (final_scores > self.threshold).astype(int), final_scores

    def end_task(self) -> None:
        encoder, decoder, config, threshold, base_model_name = self._extract_column_parts(self.base_model_factory())
        self.threshold = threshold if self.threshold is None else self.threshold
        self._base_model_name = base_model_name
        self._configs.append(config)
        self.module.add_column(encoder, decoder)

    def name(self) -> str:
        return "PNN"

    def additional_info(self) -> Dict:
        return {
            "base_model": self._base_model_name,
            "task_free": self.task_free,
            "freeze_old_columns": self.freeze_old_columns,
            "threshold": self.threshold,
            "auto_add_column": self.auto_add_column,
            "random_state": self.random_state,
            "current_task": self.current_task,
            "num_columns": self.num_columns,
            "trained_columns": len(self._trained_columns),
            "known_concepts": len(self._concept_to_task),
        }
