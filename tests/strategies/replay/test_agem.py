import logging

import numpy as np
import pytest
import torch
from torch import nn

from pyclad.models.model import Model
from pyclad.strategies.replay.agem import AGEMStrategy
from pyclad.strategies.replay.buffers.balanced import BalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayOnlyStrategy
from tests.strategies.baselines.mock_model import MockModel


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class TinyTorchModel(Model):
    def __init__(self):
        self.module = TinyModule()

    def fit(self, data: np.ndarray):
        raise AssertionError("AGEMStrategy should not delegate training to model.fit().")

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "TinyTorchModel"


class DotProductModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, x):
        return x


class DotProductModel(Model):
    def __init__(self):
        self.module = DotProductModule()

    def fit(self, data: np.ndarray):
        raise AssertionError("AGEMStrategy should not delegate training to model.fit().")

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "DotProductModel"


def test_agem_strategy_implements_replay_only_interface():
    model = TinyTorchModel()
    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=8, seed=0),
        module=model.module,
        batch_size=2,
        lr=1e-2,
        epochs=1,
        device="cpu",
    )

    assert isinstance(strategy, ReplayOnlyStrategy)
    assert strategy.name() == "AGEM"


def test_agem_strategy_tracks_concepts_in_balanced_buffer():
    model = TinyTorchModel()
    buffer = BalancedReplayBuffer(max_size=8, seed=0)
    strategy = AGEMStrategy(model, buffer, module=model.module, batch_size=2, lr=1e-2, epochs=1, device="cpu")
    first_concept = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    second_concept = np.array([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32)

    strategy.learn(first_concept, concept_id="alpha")
    strategy.learn(second_concept, concept_id="beta")

    arrays = buffer.arrays()
    assert len(arrays["examples"]) == 4
    assert set(arrays["concept_indices"].tolist()) == {0, 1}

    info = strategy.additional_info()
    assert info["task_count"] == 2
    assert info["concept_count"] == 2
    assert info["batch_size"] == 2
    assert info["replay_batch_size"] == 2
    assert info["optimizer"] == "sgd"
    assert info["projection_tolerance"] == pytest.approx(1e-6)
    assert info["epochs"] == 1
    assert "replay_buffer" in info


def test_capture_gradients_is_available_as_static_utility():
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    parameter.grad = torch.tensor([[0.5, -1.5]], dtype=torch.float32)

    captured = AGEMStrategy.capture_gradients([parameter])

    assert torch.equal(captured, torch.tensor([0.5, -1.5], dtype=torch.float32))


def test_capture_gradients_rejects_multi_device_parameter_sets():
    cpu_parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    meta_parameter = nn.Parameter(torch.empty(1, device="meta"))

    with pytest.raises(ValueError, match="single device"):
        AGEMStrategy.capture_gradients([cpu_parameter, meta_parameter])


def test_agem_strategy_requires_explicit_torch_module():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        AGEMStrategy(MockModel(), module="not-a-module")


def test_agem_strategy_rejects_unknown_optimizer():
    model = TinyTorchModel()

    with pytest.raises(ValueError, match="optimizer must be 'sgd' or 'adam'"):
        AGEMStrategy(model, module=model.module, optimizer="rmsprop")


def test_agem_strategy_rejects_negative_projection_tolerance():
    model = TinyTorchModel()

    with pytest.raises(ValueError, match="projection_tolerance must be non-negative"):
        AGEMStrategy(model, module=model.module, projection_tolerance=-1e-6)


def test_agem_strategy_allows_distinct_replay_batch_size():
    model = TinyTorchModel()
    strategy = AGEMStrategy(model, module=model.module, batch_size=4, replay_batch_size=2)

    assert strategy.additional_info()["batch_size"] == 4
    assert strategy.additional_info()["replay_batch_size"] == 2


def test_should_project_respects_tolerance():
    current = torch.tensor([1.0, 1.0], dtype=torch.float32)
    reference = torch.tensor([-1.0, 1.0 - 5e-7], dtype=torch.float32)

    assert not AGEMStrategy.should_project(current, reference, tolerance=1e-6)
    assert AGEMStrategy.should_project(current, reference, tolerance=1e-8)


def test_project_gradient_returns_current_for_non_finite_reference_norm():
    current = torch.tensor([1.0, 2.0], dtype=torch.float32)
    reference = torch.tensor([float("nan"), 0.0], dtype=torch.float32)

    projected = AGEMStrategy.project_gradient(current, reference)

    assert torch.equal(projected, current)


def test_prepare_data_reuses_identity_transform_storage():
    model = TinyTorchModel()
    strategy = AGEMStrategy(model, module=model.module, device="cpu")
    data = np.array([[1.0, 2.0]], dtype=np.float32)

    prepared = strategy._prepare_data(data)

    assert np.shares_memory(prepared.numpy(), data)


def test_should_project_warns_on_non_finite_dot_product(caplog):
    current = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    reference = torch.tensor([1.0, 1.0], dtype=torch.float32)

    with caplog.at_level(logging.WARNING):
        should_project = AGEMStrategy.should_project(current, reference)

    assert not should_project
    assert "gradient dot product is non-finite" in caplog.text


def test_agem_strategy_warns_that_adam_voids_standard_agem_guarantee(caplog):
    model = TinyTorchModel()
    strategy = AGEMStrategy(
        model,
        module=model.module,
        optimizer="adam",
        batch_size=1,
        epochs=1,
        device="cpu",
        shuffle=False,
    )

    with caplog.at_level(logging.WARNING):
        strategy._fit(np.array([[1.0, 2.0]], dtype=np.float32))

    assert "does not preserve the standard A-GEM guarantee" in caplog.text
    assert "adaptive moment estimates" in caplog.text


def test_agem_strategy_skips_empty_replay_sampling(monkeypatch):
    model = TinyTorchModel()
    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=4, seed=0),
        module=model.module,
        batch_size=1,
        epochs=1,
        device="cpu",
        shuffle=False,
    )

    def fail_if_called(batch_size: int) -> np.ndarray:
        raise AssertionError(f"_sample_replay_examples should not be called for empty buffers, got {batch_size}")

    monkeypatch.setattr(strategy, "_sample_replay_examples", fail_if_called)

    strategy._fit(np.array([[1.0, 2.0]], dtype=np.float32))


def test_agem_strategy_adam_state_tracks_only_the_restored_projected_gradient(monkeypatch):
    model = DotProductModel()
    buffer = BalancedReplayBuffer(max_size=4, seed=0)
    buffer.add(
        examples=np.array([[-1.0, 1.0]], dtype=np.float32),
        concept_indices=np.array([0], dtype=np.int64),
    )

    def linear_loss(module: DotProductModule, batch: torch.Tensor) -> torch.Tensor:
        return torch.dot(module.weights, batch.mean(dim=0))

    captured = {}
    original_adam = torch.optim.Adam

    class CaptureAdam(original_adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["optimizer"] = self

    monkeypatch.setattr(torch.optim, "Adam", CaptureAdam)

    strategy = AGEMStrategy(
        model,
        buffer,
        module=model.module,
        loss_fn=linear_loss,
        batch_size=1,
        replay_batch_size=1,
        lr=1e-3,
        optimizer="adam",
        epochs=1,
        device="cpu",
        shuffle=False,
    )

    strategy._fit(np.array([[1.0, 0.0]], dtype=np.float32))

    optimizer = captured["optimizer"]
    state = optimizer.state[model.module.weights]
    expected_projected_grad = torch.tensor([0.5, 0.5], dtype=torch.float32)

    assert torch.allclose(state["exp_avg"], 0.1 * expected_projected_grad)
    assert torch.allclose(state["exp_avg_sq"], 0.001 * expected_projected_grad.square())


def test_agem_strategy_skips_non_finite_current_gradients(caplog):
    model = TinyTorchModel()

    def nan_loss(module: TinyModule, batch: torch.Tensor) -> torch.Tensor:
        del batch
        return module.linear.weight.sum() * torch.tensor(float("nan"))

    strategy = AGEMStrategy(
        model,
        module=model.module,
        loss_fn=nan_loss,
        batch_size=1,
        epochs=1,
        device="cpu",
        shuffle=False,
    )
    weight_before = model.module.linear.weight.detach().clone()
    bias_before = model.module.linear.bias.detach().clone()

    with caplog.at_level(logging.WARNING):
        strategy._fit(np.array([[1.0, 2.0]], dtype=np.float32))

    assert "current gradient contains non-finite values" in caplog.text
    assert torch.equal(model.module.linear.weight.detach(), weight_before)
    assert torch.equal(model.module.linear.bias.detach(), bias_before)
