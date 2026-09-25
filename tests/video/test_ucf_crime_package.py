"""Focused public-package tests for UCF-Crime and COMMAND."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


class TestUcfCrimePackage:

    def setup_method(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        relative_paths = (
            "Abuse/Abuse001_x264.mp4",
            "Arrest/Arrest001_x264.mp4",
            "Normal_Videos_event/Normal001_x264.mp4",
            "Abuse/Abuse002_x264.mp4",
            "Normal_Videos_event/Normal002_x264.mp4",
        )
        for stream in ("all_rgbs", "all_flows"):
            for relative_path in relative_paths:
                path = self.root / stream / f"{relative_path}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.ones((2, 1024), dtype=np.float32))
        (self.root / "train_normal.txt").write_text(
            "Normal_Videos_event/Normal001_x264.mp4\nNormal_Videos_event/Normal001_x264.mp4\n", encoding="utf-8"
        )
        (self.root / "train_anomaly.txt").write_text(
            "Abuse/Abuse001_x264.mp4\nArrest/Arrest001_x264.mp4\n", encoding="utf-8"
        )
        (self.root / "test_normalv2.txt").write_text("Normal_Videos_event/Normal002_x264.mp4 64 -1\n", encoding="utf-8")
        (self.root / "test_anomalyv2.txt").write_text("Abuse/Abuse002_x264.mp4|64|[33, 48]\n", encoding="utf-8")

    def teardown_method(self):
        self.temporary.cleanup()

    def test_primary_scenario_is_labeled_as_command_paper_recreation(self):
        from pyclad.video.ucf_crime.scenarios import build_command_ucf_crime_scenario

        scenario = build_command_ucf_crime_scenario()
        assert scenario.protocol == "command-paper-recreation-4-4-5"
        assert scenario.task_sizes == (4, 4, 5)
        assert [task.name for task in scenario.tasks] == ["T1", "T2", "T3"]
        assert "COMMAND paper recreation" in scenario.display_name
        assert scenario.provenance["relationship"] == "clean-room recreation"
        assert scenario.provenance["source_section"] == "Section IV-C"
        assert scenario.provenance["source_doi"] == "10.1016/j.neucom.2026.132943"
        historical_alias = build_command_ucf_crime_scenario("paper-4-4-5")
        assert historical_alias.as_dict() == scenario.as_dict()

    def test_classwise_and_held_out_scenarios_cover_all_classes_once(self):
        from pyclad.video.datasets.command_ucf_crime import (
            COMMAND_UCF_CRIME_CONCEPT_ORDER,
        )
        from pyclad.video.ucf_crime.scenarios import build_command_ucf_crime_scenario

        classwise = build_command_ucf_crime_scenario("classwise")
        held_out = build_command_ucf_crime_scenario("12-1", held_out_class="Shooting")
        assert classwise.task_sizes == (1,) * 13
        assert held_out.task_sizes == (12, 1)
        assert held_out.tasks[-1].anomaly_classes == ("Shooting",)
        for scenario in (classwise, held_out):
            flattened = [name for task in scenario.tasks for name in task.anomaly_classes]
            assert set(flattened) == set(COMMAND_UCF_CRIME_CONCEPT_ORDER)
            assert len(flattened) == 13

    def test_audit_reports_duplicate_records_without_marking_archive_invalid(self):
        from pyclad.video.ucf_crime.audit import audit_command_ucf_crime

        audit = audit_command_ucf_crime(self.root, check_feature_arrays=True)
        assert audit.ready
        assert not audit.official_split_counts_match
        assert not audit.command_release_counts_match
        assert audit.train_records == 4
        assert audit.train_unique_videos == 3
        assert audit.train_unique_normal_videos == 1
        assert audit.train_unique_anomaly_videos == 2
        assert audit.duplicate_training_records == ("Normal_Videos_event/Normal001_x264.mp4",)
        assert audit.feature_arrays_checked == 10
        payload = audit.as_dict()
        assert payload["official_expected_counts"]["train_videos"] == 1610
        assert payload["command_release_expected_counts"]["train_records"] == 1620
        assert "COMMAND release manifest" in payload["provenance"]["manifest_relationship"]

    def test_duplicate_split_rows_keep_distinct_bag_ids(self):
        from pyclad.video.datasets.command_ucf_crime import CommandUcfCrimeDataset

        dataset = CommandUcfCrimeDataset(self.root, concept_order=("Abuse", "Arrest"))
        concepts = dataset.training_concepts()
        normal_bag_ids = [
            float(np.unique(concept.strategy_targets["bag_id"][concept.strategy_targets["weak_label"] == 0])[0])
            for concept in concepts
        ]
        assert len(set(normal_bag_ids)) == 2

    def test_audit_distinguishes_official_videos_from_command_manifest_rows(self):
        from pyclad.video.ucf_crime.audit import CommandUcfCrimeAudit

        audit = CommandUcfCrimeAudit(
            root=str(self.root),
            modality="two",
            train_normal_records=810,
            train_anomaly_records=810,
            train_unique_normal_videos=800,
            train_unique_anomaly_videos=810,
            train_unique_videos=1610,
            test_normal_videos=150,
            test_anomaly_videos=140,
            anomaly_train_by_class={},
            anomaly_test_by_class={},
            duplicate_training_records=("Normal_Videos_event/Normal001_x264.mp4",),
            train_test_overlaps=(),
            missing_feature_files=(),
            invalid_feature_arrays=(),
            checked_feature_arrays=False,
            feature_arrays_checked=0,
        )
        assert audit.official_split_counts_match
        assert audit.command_release_counts_match
        assert audit.train_records == 1620
        assert audit.train_unique_videos == 1610

    def test_cli_makes_focused_workflow_primary(self):
        from pyclad.video.cli import _parser

        audit = _parser().parse_args(["ucf-audit", "--data-root", str(self.root)])
        command = _parser().parse_args(["ucf-command", "--data-root", str(self.root)])
        assert not audit.check_feature_arrays
        assert audit.modality == "two"
        assert command.tasks == "T1,T2,T3"
        assert command.buffer_size == 1000
        assert command.replay_batch_size == 16
        help_text = _parser().format_help()
        assert "ucf-audit" in help_text
        assert "ucf-command" in help_text
        assert "command-xd" not in help_text
        assert "command-nola" not in help_text
        assert "prism-cl-ucf" not in help_text
