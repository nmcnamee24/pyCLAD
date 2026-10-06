"""Core video data, schema, and frame-evaluation tests."""

from __future__ import annotations

import numpy as np
import pytest


class TestVideoCore:

    def test_overlapping_window_scores_map_back_to_frames(self):
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.metrics.frame_score_utils import window_scores_to_frame_scores

        windows = (VideoWindow("video", 0, 2), VideoWindow("video", 2, 4))
        frame_scores = window_scores_to_frame_scores(windows, [0.2, 0.8], {"video": 5})
        np.testing.assert_allclose(frame_scores["video"], [0.2, 0.2, 0.5, 0.8, 0.8])

    def test_frame_metrics_report_perfect_ranking(self):
        from pyclad.video.metrics.frame_average_precision import (
            FrameAveragePrecision,
        )
        from pyclad.video.metrics.frame_roc_auc import FrameRocAuc

        labels = np.asarray([0, 0, 0, 1])
        scores = np.asarray([0.0, 0.1, 0.2, 0.9])
        for metric in (FrameRocAuc(), FrameAveragePrecision()):
            assert metric.compute(scores, None, labels) == 1.0
            assert np.isnan(metric.compute(scores, None, np.zeros(4)))

    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    @pytest.mark.parametrize("missing", ["frame", "video"])
    def test_frame_metrics_reject_incomplete_prediction_coverage(self, aggregation, missing):
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.metrics.frame_score_utils import window_scores_to_frame_scores

        windows = (VideoWindow("video", 0, 1), VideoWindow("video", 3, 4))
        frame_counts = {"video": 5}
        if missing == "video":
            windows = (VideoWindow("video", 0, 2), VideoWindow("video", 3, 4))
            frame_counts["unpredicted"] = 5
        with pytest.raises(ValueError, match="do not cover all frames"):
            window_scores_to_frame_scores(windows, [0.2, 0.8], frame_counts, aggregation=aggregation)
