"""Focused CLI and reproducibility tests."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np


class TestVideoCli:

    def test_primary_command_defaults_match_recreation_contract(self):
        from pyclad.video.cli import _parser

        arguments = _parser().parse_args(["ucf-command", "--data-root", "/tmp/ucf"])
        assert arguments.tasks == "T1,T2,T3"
        assert arguments.epochs == 100
        assert (arguments.batch_size, arguments.replay_batch_size) == (32, 16)
        assert arguments.buffer_size == 1000
        assert arguments.secondary_memory_learning_rate == 1e-05
        assert arguments.seed == 42

    def test_historical_command_paper_alias_remains_callable(self):
        from pyclad.video.cli import main

        with mock.patch("pyclad.video.cli._run_command_paper") as run:
            main(["command-paper", "--data-root", "/tmp/ucf"])
        assert run.call_args.args[0].command == "ucf-command"

    def test_top_level_help_only_advertises_supported_workflows(self):
        from pyclad.video.cli import _parser

        help_text = _parser().format_help()
        assert "ucf-audit" in help_text
        assert "ucf-command" in help_text
        assert "command-paper" not in help_text
        assert "command-nola" not in help_text
        assert "prism" not in help_text.lower()
        assert "xd-violence" not in help_text.lower()

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
            arguments = argparse.Namespace(command="test", seed=42, output_json=output, data_root=Path("/tmp/data"))
            with mock.patch.dict("os.environ", {"PYCLAD_COMMIT_SHA": "abc1234"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    _emit_json({"metrics": {"snr": float("inf")}}, arguments)
            result = json.loads(output.read_text(encoding="utf-8"))
        assert result["run"]["commit_sha"] == "abc1234"
        assert result["run"]["seed"] == 42
        assert result["run"]["arguments"]["data_root"] == "/tmp/data"
        assert result["metrics"]["snr"] is None
        assert not result["validation"]["finite"]
        assert result["validation"]["non_finite_values"] == ["$.metrics.snr"]
