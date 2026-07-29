"""NOLA stage, prepared-test, and preprocessing tests."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_prepared_video(directory: Path, *, frame_count: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    annotations = []
    tracks = []
    for frame_id in range(1, frame_count + 1):
        annotations.append(
            {
                "frame_id": frame_id,
                "filename": f"frame-{frame_id}.png",
                "objects": [
                    {
                        "class_id": 2,
                        "name": "car",
                        "relative_coordinates": {
                            "center_x": 0.5,
                            "center_y": 0.5,
                            "width": 0.2,
                            "height": 0.2,
                        },
                        "confidence": 0.9,
                    },
                    {
                        "class_id": 0,
                        "name": "person",
                        "relative_coordinates": {
                            "center_x": 0.2,
                            "center_y": 0.2,
                            "width": 0.1,
                            "height": 0.1,
                        },
                        "confidence": 0.8,
                    },
                ],
            }
        )
        tracks.append([frame_id, "1", "car", (10 + frame_id, 20, 30 + frame_id, 40)])
    (directory / f"{directory.name}.json").write_text(json.dumps(annotations), encoding="utf-8")
    np.save(directory / "tracks.npy", np.asarray(tracks, dtype=object))


class NolaDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "NOLA"
        _write_prepared_video(self.root / "Train" / "M-Train" / "mon_4_1")

    def tearDown(self):
        self.temporary.cleanup()

    def test_continual_stage_becomes_regular_video_concept(self):
        from pyclad.video import NolaContinualDataset

        dataset = NolaContinualDataset(
            self.root,
            frame_stride=1,
            stage_order=("M-Train",),
        )
        concept = dataset.training_concepts()[0]

        self.assertEqual(concept.name, "M-Train")
        self.assertEqual(concept.features.shape, (3, 8))
        self.assertEqual(concept.strategy_matrix().shape, (3, 8))
        np.testing.assert_array_equal(concept.features[:, 5:8], [[1, 1, 4]] * 3)

    def test_paper_dataset_adds_class_counts_context_and_trajectory_slot(self):
        from pyclad.video import (
            DARKNET_COCO_CLASSES,
            NOLA_PAPER_FEATURE_DIM,
            NolaPaperContinualDataset,
        )

        dataset = NolaPaperContinualDataset(
            self.root,
            frame_stride=1,
            stage_order=("M-Train",),
        )
        concept = dataset.training_concepts()[0]

        self.assertEqual(concept.features.shape, (3, NOLA_PAPER_FEATURE_DIM))
        counts = concept.features[0, 5 : 5 + len(DARKNET_COCO_CLASSES)]
        self.assertEqual(counts[DARKNET_COCO_CLASSES.index("person")], 1.0)
        self.assertEqual(counts[DARKNET_COCO_CLASSES.index("car")], 1.0)
        context = concept.features[0, 5 + len(DARKNET_COCO_CLASSES) : -1]
        np.testing.assert_array_equal(context, [2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(concept.features[0, -1], 0.0)

    def test_prepared_test_dataset_uses_nola_ground_truth(self):
        from pyclad.video import NolaPreparedTestDataset

        processed = Path(self.temporary.name) / "processed"
        video = processed / "mon_4_1"
        _write_prepared_video(video)
        (video / "metadata.json").write_text(json.dumps({"source_frame_count": 3}), encoding="utf-8")
        ground_truth = Path(self.temporary.name) / "gt.txt"
        ground_truth.write_text("mon_4_1,2,3,-1,-1\n", encoding="utf-8")

        dataset = NolaPreparedTestDataset(processed, ground_truth, frame_stride=1)

        self.assertEqual(dataset.feature_matrix().shape, (3, 8))
        np.testing.assert_array_equal(dataset.frame_labels()["mon_4_1"], [0, 1, 1])
        self.assertEqual(dataset.anomaly_intervals["mon_4_1"], ((1, 3),))

    def test_prepared_test_dataset_prefers_decoded_frame_count(self):
        from pyclad.video import NolaPreparedTestDataset

        processed = Path(self.temporary.name) / "processed-decoded"
        video = processed / "mon_4_1"
        _write_prepared_video(video)
        (video / "metadata.json").write_text(
            json.dumps({"source_frame_count": 4, "decoded_frame_count": 3}),
            encoding="utf-8",
        )
        ground_truth = Path(self.temporary.name) / "gt-decoded.txt"
        ground_truth.write_text("mon_4_1,2,3,-1,-1\n", encoding="utf-8")

        dataset = NolaPreparedTestDataset(processed, ground_truth, frame_stride=1)

        self.assertEqual(len(dataset.frame_labels()["mon_4_1"]), 3)

    def test_nola_runner_resets_and_reports_apd(self):
        from pyclad.strategies.baselines.naive import NaiveStrategy
        from pyclad.video import (
            NolaBenchmarkRunner,
            NolaContinualDataset,
            NolaPreparedTestDataset,
            NolaVideoModel,
        )

        processed = Path(self.temporary.name) / "processed-runner"
        video = processed / "mon_4_1"
        _write_prepared_video(video)
        (video / "metadata.json").write_text(json.dumps({"source_frame_count": 3}), encoding="utf-8")
        ground_truth = Path(self.temporary.name) / "gt-runner.txt"
        ground_truth.write_text("mon_4_1,2,3,-1,-1\n", encoding="utf-8")

        continual = NolaContinualDataset(
            self.root,
            frame_stride=1,
            stage_order=("M-Train",),
        )
        train = continual.training_concepts()
        test = NolaPreparedTestDataset(processed, ground_truth, frame_stride=1)
        strategy = NaiveStrategy(
            NolaVideoModel(
                layout=continual.layout,
                neighbors=1,
                apply_odit=False,
            )
        )
        result = NolaBenchmarkRunner().run(test, strategy, train_concepts=train)

        self.assertEqual(result.strategy_name, "Naive")
        self.assertEqual(set(result.frame_scores), {"mon_4_1"})
        self.assertTrue(np.isfinite(result.window_scores["mon_4_1"]).all())
        self.assertGreaterEqual(result.average_precision_delay.score, 0.0)


@unittest.skipUnless(importlib.util.find_spec("cv2") is not None, "OpenCV is optional")
class NolaPreprocessingTest(unittest.TestCase):
    def test_darknet_detector_keeps_zero_based_coco_class_ids(self):
        from pyclad.video import DarknetNolaDetector

        class FakeDetectionModel:
            @staticmethod
            def detect(*_args, **_kwargs):
                return (
                    np.asarray([0, 2]),
                    np.asarray([0.9, 0.8]),
                    np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]),
                )

        detector = DarknetNolaDetector.__new__(DarknetNolaDetector)
        detector._model = FakeDetectionModel()
        detector._classes = ("person", "bicycle", "car")
        detector.confidence_threshold = 0.25
        detector.nms_threshold = 0.45

        detections = detector(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertEqual([detection.name for detection in detections], ["person", "car"])
        self.assertEqual([detection.class_id for detection in detections], [0, 2])

    def test_native_darknet_json_is_converted_to_pixel_boxes(self):
        from pyclad.video.preprocessing.nola import _darknet_json_detection

        detection = _darknet_json_detection(
            {
                "class_id": 0,
                "name": "person",
                "confidence": 0.9,
                "relative_coordinates": {
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "width": 0.25,
                    "height": 0.5,
                },
            },
            1280,
            720,
        )

        self.assertEqual(detection.name, "person")
        self.assertEqual(detection.class_id, 0)
        np.testing.assert_allclose(detection.box, [480.0, 180.0, 800.0, 540.0])

    def test_video_preprocessing_writes_reusable_detection_and_track_cache(self):
        import cv2

        from pyclad.video import NolaDetection, preprocess_nola_video

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "mon_4_1.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (32, 24),
            )
            for value in (0, 64, 128):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()

            def detector(_):
                return (NolaDetection("car", 0.9, (2.0, 3.0, 12.0, 13.0)),)

            output = preprocess_nola_video(
                video_path,
                root / "mon_4_1",
                detector,
            )
            tracks = np.load(output / "tracks.npy", allow_pickle=True)
            annotations = json.loads((output / "mon_4_1.json").read_text(encoding="utf-8"))
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(tracks.shape, (3, 4))
            self.assertEqual(len(set(tracks[:, 1])), 1)
            self.assertEqual(len(annotations), 3)
            self.assertEqual(metadata["processed_frames"], 3)
            self.assertEqual(metadata["decoded_frame_count"], 3)
            self.assertEqual(metadata["tracker"]["name"], "simple-class-aware-iou")


if __name__ == "__main__":
    unittest.main()
