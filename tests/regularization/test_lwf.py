import numpy as np
import pytest

from pyclad.strategies.regularization.lwf import LwFStrategy
from tests.regularization.shared_models import NonTorchModel, TinyTorchModel


def test_lwf_strategy_first_task_uses_fit_and_clones_old_model():
    model = TinyTorchModel()
    strategy = LwFStrategy(model, alpha=0.3, distill_mode="latent")
    data = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

    strategy.learn(data)

    info = strategy.additional_info()
    assert strategy.name() == "LwF"
    assert model.fit_calls == 1
    assert info["model"] == "TinyTorchModel"
    assert info["alpha"] == 0.3
    assert info["distill_mode"] == "latent"
    assert info["task_count"] == 1
    assert info["has_old_model"] is True
    assert strategy._old_model is not None
    assert all(not parameter.requires_grad for parameter in strategy._old_model.module.parameters())


def test_lwf_strategy_second_task_uses_distillation_for_torch_models():
    model = TinyTorchModel()
    strategy = LwFStrategy(model, distill_mode="latent")
    first = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    second = np.array([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32)

    strategy.learn(first)
    strategy.learn(second)

    assert model.fit_calls == 1
    assert strategy.additional_info()["task_count"] == 2
    assert strategy.additional_info()["has_old_model"] is True


def test_lwf_strategy_falls_back_to_fit_for_non_torch_models():
    model = NonTorchModel()
    strategy = LwFStrategy(model, distill_mode="latent")
    data = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

    strategy.learn(data)
    strategy.learn(data)

    assert model.fit_calls == 2
    assert strategy.additional_info()["task_count"] == 2
    assert strategy.additional_info()["has_old_model"] is True


def test_lwf_strategy_rejects_invalid_distillation_mode():
    with pytest.raises(ValueError, match="Invalid distill_mode"):
        LwFStrategy(TinyTorchModel(), distill_mode="unknown")
