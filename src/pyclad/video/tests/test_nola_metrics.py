"""NOLA Average Precision-Delay tests."""

from __future__ import annotations

import unittest

import numpy as np


class AveragePrecisionDelayTest(unittest.TestCase):
    def test_threshold_sweep_returns_bounded_curve_and_area(self):
        from pyclad.video import compute_average_precision_delay

        result = compute_average_precision_delay(
            {
                "anomalous": [0.0, 0.0, 1.0, 2.0, 0.0],
                "normal": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            {"anomalous": [(2, 4)]},
            thresholds=[2.5, 1.5, 0.5, -0.5],
            maximum_delay=5,
        )

        self.assertEqual(result.thresholds.shape, (4,))
        np.testing.assert_allclose(result.normalized_delays, [1.0, 0.2, 0.0, 0.0])
        np.testing.assert_allclose(result.precisions, [0.0, 1.0, 1.0, 1.0 / 3.0])
        self.assertTrue(((result.precisions >= 0) & (result.precisions <= 1)).all())
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_interval_stop_is_clipped_to_decodable_score_length(self):
        from pyclad.video import compute_average_precision_delay

        result = compute_average_precision_delay(
            {"truncated": [0.0, 0.0, 1.0, 2.0]},
            {"truncated": [(2, 5)]},
            thresholds=[2.5, 1.5, 0.5],
            maximum_delay=4,
        )

        self.assertEqual(result.thresholds.shape, (3,))
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
