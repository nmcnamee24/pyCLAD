"""Focused CLI and reproducibility tests."""

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
    def test_primary_command_defaults_match_recreation_contract(self):
        from pyclad.video.cli import _parser

        arguments = _parser().parse_args(["ucf-command", "--data-root", "/tmp/ucf"])

        self.assertEqual(arguments.tasks, "T1,T2,T3")
        self.assertEqual(arguments.epochs, 100)
        self.assertEqual((arguments.batch_size, arguments.replay_batch_size), (32, 16))
        self.assertEqual(arguments.buffer_size, 1_000)
        self.assertEqual(arguments.secondary_memory_learning_rate, 1e-5)
        self.assertEqual(arguments.seed, 42)

    def test_historical_command_paper_alias_remains_callable(self):
        from pyclad.video.cli import _parser

        primary = _parser().parse_args(["ucf-command", "--data-root", "/tmp/ucf"])
        alias = _parser().parse_args(["command-paper", "--data-root", "/tmp/ucf"])

        primary_values = vars(primary).copy()
        alias_values = vars(alias).copy()
        primary_values.pop("command")
        alias_values.pop("command")
        self.assertEqual(alias_values, primary_values)

    def test_top_level_help_only_advertises_supported_workflows(self):
        from pyclad.video.cli import _parser

        help_text = _parser().format_help()

        self.assertIn("ucf-audit", help_text)
        self.assertIn("ucf-command", help_text)
        self.assertNotIn("command-paper", help_text)
        self.assertNotIn("command-nola", help_text)
        self.assertNotIn("prism", help_text.lower())
        self.assertNotIn("xd-violence", help_text.lower())

    def test_global_seed_reproduces_numpy_values(self):
        from pyclad.video.cli import _set_global_seed

        _set_global_seed(42)
        first = np.random.random(5)
        _set_global_seed(42)
        second = np.random.random(5)

        np.testing.assert_array_equal(first, second)

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
        self.assertEqual(result["validation"]["non_finite_values"], ["$.metrics.snr"])


if __name__ == "__main__":
    unittest.main()
