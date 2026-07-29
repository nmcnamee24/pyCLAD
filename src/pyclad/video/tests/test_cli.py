"""Reproducible CLI output and HPC helper tests."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class VideoCliTest(unittest.TestCase):
    def test_command_research_defaults_and_reproducibility_options(self):
        from pyclad.video.cli import _parser

        arguments = _parser().parse_args(["command", "--data-root", "/tmp/command"])

        self.assertEqual(arguments.hidden_dim, 128)
        self.assertEqual(arguments.embedding_dim, 128)
        self.assertEqual(arguments.memory_size, 64)
        self.assertEqual(arguments.seed, 42)
        self.assertIsNone(arguments.output_json)

    def test_global_seed_reproduces_numpy_values(self):
        from pyclad.video.cli import _set_global_seed

        _set_global_seed(42)
        first = np.random.random(5)
        _set_global_seed(42)
        second = np.random.random(5)

        np.testing.assert_array_equal(first, second)

    def test_training_video_counts_distinguish_records_from_unique_ids(self):
        from pyclad.video.cli import (
            _training_video_records,
            _unique_training_video_ids,
        )
        from pyclad.video.data import VideoFeatureConcept, VideoWindow

        concepts = (
            VideoFeatureConcept(
                name="stage-a",
                features=np.zeros((2, 1), dtype=np.float32),
                windows=(
                    VideoWindow(
                        "shared",
                        start_frame=0,
                        end_frame=0,
                        feature_index=0,
                        split="train",
                        payload={"record_index": 0},
                    ),
                    VideoWindow(
                        "shared",
                        start_frame=1,
                        end_frame=1,
                        feature_index=1,
                        split="train",
                        payload={"record_index": 1},
                    ),
                ),
            ),
            VideoFeatureConcept(
                name="stage-b",
                features=np.zeros((1, 1), dtype=np.float32),
                windows=(
                    VideoWindow(
                        "shared",
                        start_frame=0,
                        end_frame=0,
                        feature_index=0,
                        split="train",
                    ),
                ),
            ),
        )

        self.assertEqual(_training_video_records(concepts), 3)
        self.assertEqual(_unique_training_video_ids(concepts), 1)

    def test_json_output_is_atomic_standard_json_and_flags_non_finite_metrics(self):
        from pyclad.video.cli import _emit_json

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "result.json"
            arguments = argparse.Namespace(
                command="test",
                seed=42,
                output_json=output,
                data_root=Path("/tmp/data"),
            )
            with mock.patch.dict("os.environ", {"PYCLAD_COMMIT_SHA": "abc1234"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    _emit_json({"metrics": {"snr": float("inf")}}, arguments)

            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["run"]["commit_sha"], "abc1234")
        self.assertEqual(result["run"]["seed"], 42)
        self.assertEqual(result["run"]["arguments"]["data_root"], "/tmp/data")
        self.assertIsNone(result["metrics"]["snr"])
        self.assertFalse(result["validation"]["finite"])
        self.assertEqual(
            result["validation"]["non_finite_values"],
            ["$.metrics.snr"],
        )


class NolaCacheValidationTest(unittest.TestCase):
    def test_complete_stride_one_cache_is_valid(self):
        from pyclad.video.hpc.validate_nola_cache import validate_nola_cache

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "mon_4_1"
            directory.mkdir()
            (directory / "mon_4_1.json").write_text(
                json.dumps(
                    [
                        {"frame_id": 1, "objects": []},
                        {"frame_id": 2, "objects": []},
                        {"frame_id": 3, "objects": []},
                    ]
                ),
                encoding="utf-8",
            )
            np.save(directory / "tracks.npy", np.empty((0, 4), dtype=object))
            (directory / "Names.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "source_frame_count": 3,
                        "processed_frames": 3,
                        "frame_stride": 1,
                    }
                ),
                encoding="utf-8",
            )

            result = validate_nola_cache(directory)

        self.assertTrue(result["valid"])
        self.assertEqual(result["processed_frames"], 3)

    def test_partial_cache_is_rejected(self):
        from pyclad.video.hpc.validate_nola_cache import validate_nola_cache

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "mon_4_1"
            directory.mkdir()
            (directory / "mon_4_1.json").write_text("[]", encoding="utf-8")
            np.save(directory / "tracks.npy", np.empty((0, 4), dtype=object))
            (directory / "Names.txt").write_text("", encoding="utf-8")
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "source_frame_count": 3,
                        "processed_frames": 0,
                        "frame_stride": 1,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 3"):
                validate_nola_cache(directory)


if __name__ == "__main__":
    unittest.main()
