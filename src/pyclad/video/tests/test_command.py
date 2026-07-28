"""COMMAND model and strategy compatibility tests."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "COMMAND requires the optional torch dependency")
class CommandVideoModelTest(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        self.features = rng.normal(size=(10, 4)).astype(np.float32)
        self.labels = np.concatenate([np.zeros(5), np.ones(5)]).astype(np.float32)

    def _model_and_matrix(self):
        from pyclad.video import CommandVideoModel, VideoStrategySchema

        schema = VideoStrategySchema(4, target_names=("weak_label", "bag_id"))
        matrix = schema.pack(
            self.features,
            {
                "weak_label": self.labels,
                "bag_id": np.repeat([0.0, 1.0], 5),
            },
        )
        model = CommandVideoModel(
            4,
            strategy_schema=schema,
            hidden_dim=8,
            embedding_dim=6,
            memory_size=4,
            epochs=1,
            batch_size=5,
        )
        return model, matrix

    def test_regular_fit_predict_contract(self):
        from pyclad.strategies.baselines.naive import NaiveStrategy

        model, matrix = self._model_and_matrix()
        NaiveStrategy(model).learn(matrix)
        prediction = model.predict(self.features)

        self.assertEqual(prediction.anomaly_scores.shape, (10,))
        self.assertTrue(np.isfinite(prediction.anomaly_scores).all())
        self.assertTrue(((prediction.anomaly_scores >= 0) & (prediction.anomaly_scores <= 1)).all())

    def test_ewc_uses_unchanged_tensor_backbone_contract(self):
        from pyclad.strategies.regularization.ewc import EWCStrategy

        model, matrix = self._model_and_matrix()
        strategy = EWCStrategy(
            model,
            epochs=1,
            batch_size=5,
            fisher_batch_size=5,
        )
        strategy.learn(matrix)
        prediction = strategy.predict(self.features)

        self.assertEqual(prediction.anomaly_scores.shape, (10,))

    def test_lwf_agem_and_der_use_unchanged_tensor_contract(self):
        from pyclad.strategies.regularization.der import DerPlusPlus
        from pyclad.strategies.regularization.lwf import LwFStrategy
        from pyclad.strategies.replay.agem import AGEMStrategy
        from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer

        factories = [
            lambda model: LwFStrategy(model, epochs=1, batch_size=5),
            lambda model: AGEMStrategy(
                model,
                ReservoirBuffer(max_capacity=8),
                epochs=1,
                batch_size=5,
            ),
            lambda model: DerPlusPlus(
                model=model,
                buffer=ReservoirBuffer(max_capacity=8),
                epochs=1,
                batch_size=5,
            ),
        ]
        for factory in factories:
            with self.subTest(strategy=factory):
                model, matrix = self._model_and_matrix()
                strategy = factory(model)
                strategy.learn(matrix[:5])
                strategy.learn(matrix[5:])
                self.assertEqual(strategy.predict(self.features).anomaly_scores.shape, (10,))


if __name__ == "__main__":
    unittest.main()
