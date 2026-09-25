"""Core video data, schema, and frame-evaluation tests."""

from __future__ import annotations

import numpy as np


class TestVideoCore:

    def test_video_concept_is_a_regular_pyclad_concept(self):
        from pyclad.data.concept import Concept
        from pyclad.video.data.matrix import VideoStrategySchema
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.data.video_concept import VideoConcept

        schema = VideoStrategySchema(2, target_names=("weak_label",))
        concept = VideoConcept.from_features(
            name="T1",
            features=np.asarray([[1.0, 2.0]], dtype=np.float32),
            windows=(VideoWindow("video", 0, 0, 0, label=1),),
            strategy_schema=schema,
            strategy_targets={"weak_label": [1]},
        )
        assert isinstance(concept, Concept)
        assert concept.data.shape == (1, 3)
        np.testing.assert_array_equal(concept.features, [[1.0, 2.0]])

    def test_strategy_schema_keeps_targets_out_of_model_features(self):
        from pyclad.video.data.matrix import VideoStrategySchema

        schema = VideoStrategySchema(feature_dim=2, target_names=("weak_label", "bag_id"))
        features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        matrix = schema.pack(features, {"weak_label": [0, 1], "bag_id": [10, 11]})
        np.testing.assert_array_equal(schema.features(matrix), features)
        np.testing.assert_array_equal(schema.targets(matrix)["weak_label"], [0, 1])
        np.testing.assert_array_equal(schema.targets(matrix)["bag_id"], [10, 11])

    def test_overlapping_window_scores_map_back_to_frames(self):
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.metrics.frame import window_scores_to_frame_scores

        windows = (VideoWindow("video", 0, 2, 0), VideoWindow("video", 2, 4, 1))
        frame_scores = window_scores_to_frame_scores(windows, [0.2, 0.8], {"video": 5})
        np.testing.assert_allclose(frame_scores["video"], [0.2, 0.2, 0.5, 0.8, 0.8])

    def test_frame_metrics_report_perfect_ranking(self):
        from pyclad.video.metrics.frame import compute_video_frame_metrics

        labels = {"normal": np.asarray([0, 0]), "anomaly": np.asarray([0, 1])}
        scores = {"normal": np.asarray([0.0, 0.1]), "anomaly": np.asarray([0.2, 0.9])}
        metrics = compute_video_frame_metrics(scores, labels)
        assert metrics.auc == 1.0
        assert metrics.ap == 1.0
        assert metrics.auc_anomalous_videos == 1.0
        assert metrics.ap_anomalous_videos == 1.0
