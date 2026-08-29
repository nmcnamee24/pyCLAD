"""Core video data, schema, and frame-evaluation tests."""

from __future__ import annotations

import unittest

import numpy as np


class VideoCoreTest(unittest.TestCase):
    def test_strategy_schema_keeps_targets_out_of_model_features(self):
        from pyclad.video import VideoStrategySchema

        schema = VideoStrategySchema(feature_dim=2, target_names=("weak_label", "bag_id"))
        features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        matrix = schema.pack(
            features,
            {
                "weak_label": [0, 1],
                "bag_id": [10, 11],
            },
        )

        np.testing.assert_array_equal(schema.features(matrix), features)
        np.testing.assert_array_equal(schema.targets(matrix)["weak_label"], [0, 1])
        np.testing.assert_array_equal(schema.targets(matrix)["bag_id"], [10, 11])

    def test_overlapping_window_scores_map_back_to_frames(self):
        from pyclad.video import VideoWindow, window_scores_to_frame_scores

        windows = (
            VideoWindow("video", 0, 2, 0),
            VideoWindow("video", 2, 4, 1),
        )

        frame_scores = window_scores_to_frame_scores(
            windows,
            [0.2, 0.8],
            {"video": 5},
        )

        np.testing.assert_allclose(frame_scores["video"], [0.2, 0.2, 0.5, 0.8, 0.8])

    def test_frame_metrics_report_perfect_ranking(self):
        from pyclad.video import compute_video_frame_metrics

        labels = {
            "normal": np.asarray([0, 0]),
            "anomaly": np.asarray([0, 1]),
        }
        scores = {
            "normal": np.asarray([0.0, 0.1]),
            "anomaly": np.asarray([0.2, 0.9]),
        }

        metrics = compute_video_frame_metrics(scores, labels)

        self.assertEqual(metrics.auc, 1.0)
        self.assertEqual(metrics.ap, 1.0)
        self.assertEqual(metrics.auc_anomalous_videos, 1.0)
        self.assertEqual(metrics.ap_anomalous_videos, 1.0)


if __name__ == "__main__":
    unittest.main()
