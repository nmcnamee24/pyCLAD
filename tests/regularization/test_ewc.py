import numpy as np
import pytest

from pyclad.strategies.regularization.ewc import EWCStrategy
from tests.regularization.shared_models import NonTorchModel, TinyTorchModel


def test_ewc_strategy_tracks_saved_parameters_and_importances():
    strategy = EWCStrategy(TinyTorchModel(), ewc_lambda=25.0)
    data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)

    strategy.learn(data)

    info = strategy.additional_info()
    assert strategy.name() == "EWC"
    assert info["model"] == "TinyTorchModel"
    assert info["ewc_lambda"] == 25.0
    assert info["task_count"] == 1
    assert info["num_saved_tasks"] == 1
    assert info["total_stored_params"] > 0
    assert info["total_stored_importances"] > 0
    assert set(strategy._saved_params.keys()) == {0}
    assert set(strategy._importances.keys()) == {0}


def test_ewc_strategy_keep_latest_task_only_discards_older_statistics():
    strategy = EWCStrategy(TinyTorchModel(), keep_importance_data=False)
    first = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    second = np.array([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32)

    strategy.learn(first)
    strategy.learn(second)

    assert strategy.additional_info()["task_count"] == 2
    assert set(strategy._saved_params.keys()) == {1}
    assert set(strategy._importances.keys()) == {1}


def test_ewc_strategy_requires_torch_backed_model():
    strategy = EWCStrategy(NonTorchModel())

    with pytest.raises(TypeError, match="PyTorch-backed model"):
        strategy.learn(np.array([[1.0, 2.0]], dtype=np.float32))


def test_ewc_strategy_rejects_unimplemented_online_mode():
    with pytest.raises(NotImplementedError, match="Online mode not yet implemented"):
        EWCStrategy(TinyTorchModel(), mode="online")
