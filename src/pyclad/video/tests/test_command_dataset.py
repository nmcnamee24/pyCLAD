"""COMMAND UCF-Crime archive adapter tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class CommandUcfCrimeDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for stream in ("all_rgbs", "all_flows"):
            for category in ("Abuse", "Arrest", "Normal_Videos_event"):
                (self.root / stream / category).mkdir(parents=True, exist_ok=True)

        train_anomaly = (
            "Abuse/Abuse001_x264.mp4\n"
            "Arrest/Arrest001_x264.mp4\n"
        )
        train_normal = (
            "Normal_Videos_event/Normal001_x264.mp4\n"
            "Normal_Videos_event/Normal002_x264.mp4\n"
        )
        (self.root / "train_anomaly.txt").write_text(train_anomaly, encoding="utf-8")
        (self.root / "train_normal.txt").write_text(train_normal, encoding="utf-8")
        (self.root / "test_anomalyv2.txt").write_text(
            "Abuse/Abuse002_x264.mp4|64|[33, 48]\n",
            encoding="utf-8",
        )
        (self.root / "test_normalv2.txt").write_text(
            "Normal_Videos_event/Normal003_x264.mp4 64 -1\n",
            encoding="utf-8",
        )

        paths = (
            "Abuse/Abuse001_x264.mp4",
            "Abuse/Abuse002_x264.mp4",
            "Arrest/Arrest001_x264.mp4",
            "Normal_Videos_event/Normal001_x264.mp4",
            "Normal_Videos_event/Normal002_x264.mp4",
            "Normal_Videos_event/Normal003_x264.mp4",
        )
        for index, relative_path in enumerate(paths):
            rgb = np.full((2, 1024), index + 1, dtype=np.float32)
            flow = np.full((2, 1024), index + 11, dtype=np.float32)
            np.save(self.root / "all_rgbs" / f"{relative_path}.npy", rgb)
            np.save(self.root / "all_flows" / f"{relative_path}.npy", flow)

    def tearDown(self):
        self.temporary.cleanup()

    def _dataset(self):
        from pyclad.video import CommandUcfCrimeDataset

        return CommandUcfCrimeDataset(
            self.root,
            concept_order=("Abuse", "Arrest"),
        )

    def test_balanced_training_concepts_pack_weak_targets(self):
        dataset = self._dataset()
        concepts = dataset.training_concepts()

        self.assertEqual([concept.name for concept in concepts], ["Abuse", "Arrest"])
        for concept in concepts:
            self.assertEqual(concept.features.shape, (4, 2048))
            self.assertEqual(concept.strategy_matrix().shape, (4, 2050))
            self.assertEqual(
                {window.payload["record_index"] for window in concept.windows},
                {0, 1},
            )
            np.testing.assert_array_equal(
                np.sort(np.unique(concept.strategy_targets["weak_label"])),
                [0.0, 1.0],
            )
            self.assertEqual(len(np.unique(concept.strategy_targets["bag_id"])), 2)

    def test_test_windows_and_frame_labels_match_archive_intervals(self):
        dataset = self._dataset()
        concept = dataset.test_concept(max_normal_videos=1, max_anomaly_videos=1)
        labels = dataset.frame_labels()

        self.assertEqual(concept.features.shape, (4, 2048))
        self.assertEqual(len(concept.windows), 4)
        self.assertEqual(int(labels["Abuse/Abuse002_x264.mp4"].sum()), 16)
        self.assertEqual(int(labels["Normal_Videos_event/Normal003_x264.mp4"].sum()), 0)
        anomaly_windows = [
            window
            for window in concept.windows
            if window.video_id == "Abuse/Abuse002_x264.mp4"
        ]
        self.assertEqual([window.label for window in anomaly_windows], [0, 1])


if __name__ == "__main__":
    unittest.main()
