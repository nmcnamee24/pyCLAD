import numpy as np
import pytest
import torch
from torch import nn

from pyclad.models.model import Model
from pyclad.strategies.regularization.ewc import EWCStrategy
from tests.regularization.shared_models import TinyTorchModel


def make_strategy(model: TinyTorchModel, **overrides) -> EWCStrategy:
    params = {
        "module": model.module,
        "loss_fn": EWCStrategy.default_loss_fn,
        "data_transform": EWCStrategy.identity_transform,
        "batch_size": model.batch_size,
        "lr": model.module.lr,
        "epochs": model.epochs,
        "device": model.device,
        "shuffle": True,
        "fisher_estimation_mode": "eval",
        "constraint_retention": "all",
    }
    params.update(overrides)
    return EWCStrategy(model, **params)


class CancellationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.train_loss = nn.MSELoss()

        with torch.no_grad():
            self.linear.weight.fill_(0.0)
            self.linear.bias.fill_(0.0)

    def forward(self, x):
        return self.linear(x)


class CancellationModel(Model):
    def __init__(self):
        self.module = CancellationModule()
        self._auto_threshold = False

    def fit(self, data: np.ndarray):
        raise AssertionError("EWCStrategy should not delegate training to model.fit().")

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "CancellationModel"


class ModeTrackingLoss:
    def __init__(self):
        self.seen_modes = []

    def __call__(self, module: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        self.seen_modes.append(module.training)
        return torch.nn.functional.mse_loss(module(batch), batch)


def compute_naive_penalty(strategy: EWCStrategy) -> torch.Tensor:
    module = strategy._module
    penalty = next(module.parameters()).new_tensor(0.0)

    for task_id, task_params in strategy._saved_params.items():
        task_importances = strategy._importances.get(task_id, {})

        for name, param in module.named_parameters():
            if not param.requires_grad or name not in task_params or name not in task_importances:
                continue

            reference = task_params[name].to(device=param.device, dtype=param.dtype)
            importance = task_importances[name].to(device=param.device, dtype=param.dtype)
            penalty = penalty + (importance * (param - reference).pow(2)).sum()

    return 0.5 * penalty


def test_ewc_strategy_tracks_saved_parameters_and_importances():
    model = TinyTorchModel()
    strategy = make_strategy(model, ewc_lambda=25.0)
    data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)

    strategy.learn(data)

    info = strategy.additional_info()
    assert strategy.name() == "EWC"
    assert info["model"] == "TinyTorchModel"
    assert info["ewc_lambda"] == 25.0
    assert info["batch_size"] == 2
    assert info["lr"] == pytest.approx(1e-2)
    assert info["epochs"] == 1
    assert info["device"] == "cpu"
    assert info["shuffle"] is True
    assert info["fisher_estimation_mode"] == "eval"
    assert info["constraint_retention"] == "all"
    assert info["task_count"] == 1
    assert info["num_saved_tasks"] == 1
    assert info["total_stored_params"] > 0
    assert info["total_stored_importances"] > 0
    assert set(strategy._saved_params.keys()) == {0}
    assert set(strategy._importances.keys()) == {0}
    assert model.fit_calls == 0


def test_ewc_strategy_defaults_to_conservative_reconstruction_lambda():
    strategy = make_strategy(TinyTorchModel())

    assert strategy.additional_info()["ewc_lambda"] == pytest.approx(1.0)


def test_ewc_strategy_retains_all_constraints_by_default():
    model = TinyTorchModel()
    strategy = make_strategy(model)
    first = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    second = np.array([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32)

    strategy.learn(first)
    strategy.learn(second)

    assert strategy.additional_info()["task_count"] == 2
    assert set(strategy._saved_params.keys()) == {0, 1}
    assert set(strategy._importances.keys()) == {0, 1}


def test_ewc_cached_penalty_matches_naive_taskwise_penalty():
    model = TinyTorchModel()
    strategy = make_strategy(model)
    with torch.no_grad():
        for parameter in strategy._module.parameters():
            parameter.fill_(0.25)

    strategy._saved_params = {
        0: {
            name: torch.full_like(param.detach().cpu(), 0.1)
            for name, param in strategy._module.named_parameters()
            if param.requires_grad
        },
        1: {
            name: torch.full_like(param.detach().cpu(), -0.2)
            for name, param in strategy._module.named_parameters()
            if param.requires_grad
        },
    }
    strategy._importances = {
        0: {
            name: torch.full_like(param.detach().cpu(), 0.3)
            for name, param in strategy._module.named_parameters()
            if param.requires_grad
        },
        1: {
            name: torch.full_like(param.detach().cpu(), 0.7)
            for name, param in strategy._module.named_parameters()
            if param.requires_grad
        },
    }
    strategy._rebuild_penalty_cache()

    cached = strategy._compute_ewc_penalty(strategy._module)
    naive = compute_naive_penalty(strategy)

    assert cached.item() == pytest.approx(naive.item(), rel=1e-6, abs=1e-8)


def test_ewc_strategy_latest_retention_keeps_only_most_recent_constraint():
    model = TinyTorchModel()
    strategy = make_strategy(model, constraint_retention="latest")
    first = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    second = np.array([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32)

    strategy.learn(first)
    strategy.learn(second)

    assert strategy.additional_info()["task_count"] == 2
    assert set(strategy._saved_params.keys()) == {1}
    assert set(strategy._importances.keys()) == {1}


def test_ewc_strategy_requires_explicit_torch_module():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        EWCStrategy(
            TinyTorchModel(),
            module="not-a-module",
            loss_fn=EWCStrategy.default_loss_fn,
            data_transform=EWCStrategy.identity_transform,
            batch_size=2,
            lr=1e-2,
            epochs=1,
            device="cpu",
            shuffle=True,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("batch_size", 0, "batch_size must be positive"),
        ("lr", 0.0, "lr must be positive"),
        ("epochs", 0, "epochs must be positive"),
        ("fisher_estimation_mode", "invalid", "fisher_estimation_mode must be 'eval' or 'train'"),
        ("constraint_retention", "invalid", "constraint_retention must be 'all' or 'latest'"),
    ],
)
def test_ewc_strategy_validates_numeric_constructor_args(field_name, value, message):
    model = TinyTorchModel()

    with pytest.raises(ValueError, match=message):
        make_strategy(model, **{field_name: value})


def test_ewc_strategy_requires_callable_loss_fn():
    model = TinyTorchModel()

    with pytest.raises(TypeError, match="loss_fn"):
        EWCStrategy(
            model,
            module=model.module,
            loss_fn="not-callable",
            data_transform=EWCStrategy.identity_transform,
            batch_size=2,
            lr=1e-2,
            epochs=1,
            device="cpu",
            shuffle=True,
        )


def test_ewc_strategy_requires_callable_data_transform():
    model = TinyTorchModel()

    with pytest.raises(TypeError, match="data_transform"):
        EWCStrategy(
            model,
            module=model.module,
            loss_fn=EWCStrategy.default_loss_fn,
            data_transform="not-callable",
            batch_size=2,
            lr=1e-2,
            epochs=1,
            device="cpu",
            shuffle=True,
        )


def test_ewc_strategy_predict_delegates_to_wrapped_model():
    model = TinyTorchModel()
    strategy = make_strategy(model)
    data = np.array([[1.0, 2.0]], dtype=np.float32)

    predictions, scores = strategy.predict(data)

    assert model.predict_calls == 1
    np.testing.assert_array_equal(model.last_predict_data, data)
    np.testing.assert_array_equal(predictions, np.array([0]))
    np.testing.assert_array_equal(scores, np.array([0.0]))


def test_ewc_strategy_calibrates_threshold_when_enabled():
    model = TinyTorchModel()
    model._auto_threshold = True
    strategy = make_strategy(model)
    data = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

    strategy.learn(data)

    assert model.calibration_calls == 1
    np.testing.assert_array_equal(model.calibration_data, data)


def test_ewc_fisher_estimation_uses_mean_squared_per_sample_gradients():
    model = CancellationModel()
    strategy = EWCStrategy(
        model,
        module=model.module,
        loss_fn=EWCStrategy.default_loss_fn,
        data_transform=EWCStrategy.identity_transform,
        batch_size=2,
        lr=1e-2,
        epochs=1,
        device="cpu",
        shuffle=True,
        fisher_estimation_mode="eval",
        constraint_retention="all",
    )
    data = np.array([[1.0], [-1.0]], dtype=np.float32)

    fisher = strategy._compute_fisher_information(data)

    assert fisher["linear.weight"].item() == pytest.approx(4.0)
    assert fisher["linear.bias"].item() == pytest.approx(4.0)


def test_ewc_fisher_eval_mode_restores_previous_training_state():
    model = TinyTorchModel()
    loss_fn = ModeTrackingLoss()
    strategy = make_strategy(model, loss_fn=loss_fn, fisher_estimation_mode="eval")
    data = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

    model.module.train()
    fisher = strategy._compute_fisher_information(data)

    assert fisher
    assert loss_fn.seen_modes == [False, False]
    assert model.module.training is True


def test_ewc_fisher_train_mode_restores_previous_eval_state():
    model = TinyTorchModel()
    loss_fn = ModeTrackingLoss()
    strategy = make_strategy(model, loss_fn=loss_fn, fisher_estimation_mode="train")
    data = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

    model.module.eval()
    fisher = strategy._compute_fisher_information(data)

    assert fisher
    assert loss_fn.seen_modes == [True, True]
    assert model.module.training is False
