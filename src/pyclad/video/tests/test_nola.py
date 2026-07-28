"""NOLA feature, scoring, and strategy compatibility tests."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np


class NolaVideoModelTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        spatial = rng.normal(size=(24, 5))
        temporal = rng.normal(size=(24, 3))
        trajectory_error = np.abs(rng.normal(scale=0.1, size=24))

        from pyclad.video import pack_nola_features

        self.features, self.layout = pack_nola_features(spatial, temporal, trajectory_error)

    def test_novel_features_receive_larger_knn_scores(self):
        from pyclad.strategies.baselines.cumulative import CumulativeStrategy
        from pyclad.video import NolaVideoModel

        model = NolaVideoModel(
            layout=self.layout,
            neighbors=3,
            distance_aggregation="mean",
            apply_odit=False,
        )
        strategy = CumulativeStrategy(model)
        strategy.learn(self.features)
        nominal_scores = strategy.predict(self.features).anomaly_scores

        shifted = self.features.copy()
        shifted[:, :8] += 10.0
        shifted_scores = strategy.predict(shifted).anomaly_scores

        self.assertGreater(float(shifted_scores.mean()), float(nominal_scores.mean()))

    def test_odit_and_nms_utilities(self):
        from pyclad.video import non_maximum_suppression, odit_cusum

        np.testing.assert_allclose(odit_cusum([8.0, 6.0, 10.0], drift=7.0), [1.0, 0.0, 3.0])
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 9.0, 9.0],
                [20.0, 20.0, 25.0, 25.0],
            ]
        )
        kept = non_maximum_suppression(boxes, overlap_threshold=0.5, scores=[0.9, 0.8, 0.7])
        np.testing.assert_array_equal(kept, [0, 2])

    def test_reference_feature_families_can_be_built(self):
        from pyclad.video import (
            build_nola_trajectory_examples,
            nola_spatial_object_features,
            nola_temporal_object_features,
        )

        temporal = nola_temporal_object_features(
            [
                {"name": "car", "confidence": 0.9},
                {"name": "person", "confidence": 0.8},
                {"name": "person", "confidence": 0.2},
            ],
            hour=14,
        )
        np.testing.assert_array_equal(temporal, [1.0, 1.0, 14.0])

        spatial = nola_spatial_object_features(
            [[0, 0, 10, 10], [5, 5, 12, 12]],
            ["car", "truck"],
        )
        self.assertEqual(spatial.shape, (2, 5))

        track = np.arange(24 * 4, dtype=np.float32).reshape(24, 4)
        sequences, next_boxes = build_nola_trajectory_examples(
            [track],
            sequence_length=4,
            stride=5,
            frame_size=(100, 100),
        )
        self.assertEqual(sequences.shape, (4, 4, 4))
        self.assertEqual(next_boxes.shape, (4, 4))

    def test_mste_and_replay_strategies_use_regular_model_contract(self):
        from pyclad.strategies.baselines.mste import MSTE
        from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
        from pyclad.strategies.replay.replay import ReplayEnhancedStrategy, ReplayOnlyStrategy
        from pyclad.strategies.replay.selection.random import RandomSelection
        from pyclad.video import NolaVideoModel

        model_factory = lambda: NolaVideoModel(  # noqa: E731
            layout=self.layout,
            neighbors=3,
            apply_odit=False,
        )
        mste = MSTE(model_factory)
        mste.learn(self.features, concept_id="camera-1")
        self.assertEqual(
            mste.predict(self.features, concept_id="camera-1").anomaly_scores.shape,
            (24,),
        )

        for strategy_class in (ReplayOnlyStrategy, ReplayEnhancedStrategy):
            with self.subTest(strategy=strategy_class):
                buffer = AdaptiveBalancedReplayBuffer(
                    selection_method=RandomSelection(),
                    max_size=24,
                )
                strategy = strategy_class(model=model_factory(), buffer=buffer)
                strategy.learn(self.features[:12])
                strategy.learn(self.features[12:])
                self.assertEqual(strategy.predict(self.features).anomaly_scores.shape, (24,))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "trajectory predictor requires torch")
class NolaTrajectoryPredictorTest(unittest.TestCase):
    def test_predict_and_error_shapes(self):
        from pyclad.video import NolaTrajectoryPredictor

        rng = np.random.default_rng(5)
        trajectories = rng.normal(size=(6, 4, 4)).astype(np.float32)
        next_boxes = trajectories[:, -1]
        predictor = NolaTrajectoryPredictor(
            hidden_dim=4,
            layers=1,
            epochs=1,
            batch_size=3,
        )
        predictor.fit(trajectories, next_boxes)

        self.assertEqual(predictor.predict(trajectories).shape, (6, 4))
        self.assertEqual(predictor.errors(trajectories, next_boxes).shape, (6,))


if __name__ == "__main__":
    unittest.main()
