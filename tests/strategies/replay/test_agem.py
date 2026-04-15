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
        self.train_loss = nn.MSELoss()
        self.lr = 1e-2

    def forward(self, x):
        return self.linear(x)


class TinyTorchModel(Model):
    def __init__(self):
        self.module = TinyModule()
        self.epochs = 1
        self.batch_size = 2
        self.device = torch.device("cpu")

    def fit(self, data: np.ndarray):
        raise AssertionError("AGEMStrategy should not delegate training to model.fit().")

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "TinyTorchModel"


def test_agem_strategy_implements_replay_only_interface():
    strategy = AGEMStrategy(TinyTorchModel(), BalancedReplayBuffer(max_size=8, seed=0))

    assert isinstance(strategy, ReplayOnlyStrategy)
    assert strategy.name() == "AGEM"


def test_agem_strategy_tracks_concepts_in_balanced_buffer():
    model = TinyTorchModel()
    buffer = BalancedReplayBuffer(max_size=8, seed=0)
    strategy = AGEMStrategy(model, buffer)
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
    assert "replay_buffer" in info


def test_capture_gradients_is_available_as_static_utility():
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    parameter.grad = torch.tensor([[0.5, -1.5]], dtype=torch.float32)

    captured = AGEMStrategy.capture_gradients([parameter])

    assert torch.equal(captured, torch.tensor([0.5, -1.5], dtype=torch.float32))


def test_agem_strategy_requires_torch_backed_model():
    with pytest.raises(TypeError, match="PyTorch-backed model"):
        AGEMStrategy(MockModel())
