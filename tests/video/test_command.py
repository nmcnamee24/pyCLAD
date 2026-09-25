"""COMMAND model and strategy compatibility tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pyclad.models.training.runners.standard import StandardRunner

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="COMMAND requires the optional torch dependency")
class TestCommandVideoModel:

    def setup_method(self):
        import torch

        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        self.features = rng.normal(size=(10, 4)).astype(np.float32)
        self.labels = np.concatenate([np.zeros(5), np.ones(5)]).astype(np.float32)

    def _model_and_matrix(self):
        from pyclad.video.data.matrix import VideoStrategySchema
        from pyclad.video.models.command.model import CommandVideoModel

        schema = VideoStrategySchema(4, target_names=("weak_label", "bag_id"))
        matrix = schema.pack(self.features, {"weak_label": self.labels, "bag_id": np.repeat([0.0, 1.0], 5)})
        model = CommandVideoModel(4, strategy_schema=schema, hidden_dim=8, embedding_dim=6, memory_size=4)
        return (model, matrix)

    def test_regular_fit_predict_contract(self):
        from pyclad.models.adapters.torch_adapter import TorchModelAdapter
        from pyclad.models.torch_backbone import TorchBackbone
        from pyclad.strategies.baselines.naive import NaiveStrategy

        backbone, matrix = self._model_and_matrix()
        model = TorchModelAdapter(backbone, runner=StandardRunner(max_epochs=1), batch_size=5)
        NaiveStrategy(model).learn(matrix)
        prediction = model.predict(self.features)
        assert isinstance(backbone, TorchBackbone)
        assert prediction.anomaly_scores.shape == (10,)
        assert np.isfinite(prediction.anomaly_scores).all()
        assert ((prediction.anomaly_scores >= 0) & (prediction.anomaly_scores <= 1)).all()

    def test_ewc_uses_unchanged_tensor_backbone_contract(self):
        from pyclad.strategies.regularization.ewc import EWCStrategy

        model, matrix = self._model_and_matrix()
        strategy = EWCStrategy(model, runner=StandardRunner(max_epochs=1), batch_size=5, fisher_batch_size=5)
        strategy.learn(matrix)
        prediction = strategy.predict(self.features)
        assert prediction.anomaly_scores.shape == (10,)

    def test_lwf_agem_and_der_use_unchanged_tensor_contract(self):
        from pyclad.strategies.regularization.der import DerPlusPlus
        from pyclad.strategies.regularization.lwf import LwFStrategy
        from pyclad.strategies.replay.agem import AGEMStrategy
        from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer

        factories = [
            lambda model: LwFStrategy(model, runner=StandardRunner(max_epochs=1), batch_size=5),
            lambda model: AGEMStrategy(
                model, buffer=ReservoirBuffer(max_capacity=8), runner=StandardRunner(max_epochs=1), batch_size=5
            ),
            lambda model: DerPlusPlus(
                model=model, buffer=ReservoirBuffer(max_capacity=8), runner=StandardRunner(max_epochs=1), batch_size=5
            ),
        ]
        for factory in factories:
            model, matrix = self._model_and_matrix()
            strategy = factory(model)
            strategy.learn(matrix[:5])
            strategy.learn(matrix[5:])
            assert strategy.predict(self.features).anomaly_scores.shape == (10,)
