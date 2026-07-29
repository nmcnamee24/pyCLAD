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
        np.testing.assert_allclose(result.precisions, [0.0, 1.0, 1.0, 1.0 / 4.0])
        np.testing.assert_array_equal(result.true_positives, [0, 1, 1, 1])
        np.testing.assert_array_equal(result.false_positives, [0, 0, 0, 3])
        self.assertTrue(((result.precisions >= 0) & (result.precisions <= 1)).all())
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_all_intervals_and_false_alarm_runs_are_counted(self):
        from pyclad.video import compute_average_precision_delay

        result = compute_average_precision_delay(
            {
                "two-events": [1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
                "normal": [0.0] * 8,
            },
            {"two-events": [(3, 4), (7, 8)]},
            thresholds=[1.5],
            maximum_delay=8,
        )

        np.testing.assert_array_equal(result.true_positives, [2])
        np.testing.assert_array_equal(result.false_positives, [1])
        np.testing.assert_allclose(result.normalized_delays, [0.0])
        np.testing.assert_allclose(result.precisions, [2.0 / 3.0])

    def test_degenerate_scores_are_rejected_explicitly(self):
        from pyclad.video import (
            DegenerateNolaScoresError,
            require_non_degenerate_nola_scores,
        )

        with self.assertRaises(DegenerateNolaScoresError):
            require_non_degenerate_nola_scores(np.zeros(20))

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

    def test_exact_incremental_sweep_matches_explicit_threshold_states(self):
        from pyclad.video import compute_average_precision_delay

        scores = {
            "anomalous": np.asarray([0.3, 0.8, 0.1, 0.7, 0.2]),
            "normal": np.asarray([0.4, 0.6, 0.5, 0.0, 0.9]),
        }
        intervals = {"anomalous": [(1, 4)]}
        unique_scores = np.unique(np.concatenate(tuple(scores.values())))[::-1]
        explicit_thresholds = np.concatenate(
            (
                [np.nextafter(unique_scores[0], np.inf)],
                np.nextafter(unique_scores, -np.inf),
            )
        )

        exact = compute_average_precision_delay(
            scores,
            intervals,
            maximum_delay=5,
        )
        explicit = compute_average_precision_delay(
            scores,
            intervals,
            thresholds=explicit_thresholds,
            maximum_delay=5,
        )

        np.testing.assert_allclose(exact.thresholds, explicit.thresholds)
        np.testing.assert_array_equal(exact.true_positives, explicit.true_positives)
        np.testing.assert_array_equal(exact.false_positives, explicit.false_positives)
        np.testing.assert_allclose(exact.normalized_delays, explicit.normalized_delays)
        np.testing.assert_allclose(exact.precisions, explicit.precisions)
        self.assertAlmostEqual(exact.score, explicit.score)


if __name__ == "__main__":
    unittest.main()
