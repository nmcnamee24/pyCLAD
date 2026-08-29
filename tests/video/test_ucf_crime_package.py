"""Focused public-package tests for UCF-Crime and COMMAND."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class UcfCrimePackageTest(unittest.TestCase):
    def setUp(self):
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
            "Normal_Videos_event/Normal001_x264.mp4\n" "Normal_Videos_event/Normal001_x264.mp4\n",
            encoding="utf-8",
        )
        (self.root / "train_anomaly.txt").write_text(
            "Abuse/Abuse001_x264.mp4\n" "Arrest/Arrest001_x264.mp4\n",
            encoding="utf-8",
        )
        (self.root / "test_normalv2.txt").write_text(
            "Normal_Videos_event/Normal002_x264.mp4 64 -1\n",
            encoding="utf-8",
        )
        (self.root / "test_anomalyv2.txt").write_text(
            "Abuse/Abuse002_x264.mp4|64|[33, 48]\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_primary_scenario_is_labeled_as_command_paper_recreation(self):
        from pyclad.video.ucf_crime import build_command_ucf_crime_scenario

        scenario = build_command_ucf_crime_scenario()

        self.assertEqual(scenario.protocol, "command-paper-recreation-4-4-5")
        self.assertEqual(scenario.task_sizes, (4, 4, 5))
        self.assertEqual([task.name for task in scenario.tasks], ["T1", "T2", "T3"])
        self.assertIn("COMMAND paper recreation", scenario.display_name)
        self.assertEqual(scenario.provenance["relationship"], "clean-room recreation")
        self.assertEqual(scenario.provenance["source_section"], "Section IV-C")
        self.assertEqual(scenario.provenance["source_doi"], "10.1016/j.neucom.2026.132943")

        historical_alias = build_command_ucf_crime_scenario("paper-4-4-5")
        self.assertEqual(historical_alias.as_dict(), scenario.as_dict())

    def test_classwise_and_held_out_scenarios_cover_all_classes_once(self):
        from pyclad.video.ucf_crime import (
            COMMAND_UCF_CRIME_CONCEPT_ORDER,
            build_command_ucf_crime_scenario,
        )

        classwise = build_command_ucf_crime_scenario("classwise")
        held_out = build_command_ucf_crime_scenario("12-1", held_out_class="Shooting")

        self.assertEqual(classwise.task_sizes, (1,) * 13)
        self.assertEqual(held_out.task_sizes, (12, 1))
        self.assertEqual(held_out.tasks[-1].anomaly_classes, ("Shooting",))
        for scenario in (classwise, held_out):
            flattened = [name for task in scenario.tasks for name in task.anomaly_classes]
            self.assertEqual(set(flattened), set(COMMAND_UCF_CRIME_CONCEPT_ORDER))
            self.assertEqual(len(flattened), 13)

    def test_audit_reports_duplicate_records_without_marking_archive_invalid(self):
        from pyclad.video.ucf_crime import audit_command_ucf_crime

        audit = audit_command_ucf_crime(self.root, check_feature_arrays=True)

        self.assertTrue(audit.ready)
        self.assertFalse(audit.official_split_counts_match)
        self.assertFalse(audit.command_release_counts_match)
        self.assertEqual(audit.train_records, 4)
        self.assertEqual(audit.train_unique_videos, 3)
        self.assertEqual(audit.train_unique_normal_videos, 1)
        self.assertEqual(audit.train_unique_anomaly_videos, 2)
        self.assertEqual(
            audit.duplicate_training_records,
            ("Normal_Videos_event/Normal001_x264.mp4",),
        )
        self.assertEqual(audit.feature_arrays_checked, 10)
        payload = audit.as_dict()
        self.assertEqual(payload["official_expected_counts"]["train_videos"], 1_610)
        self.assertEqual(payload["command_release_expected_counts"]["train_records"], 1_620)
        self.assertIn("COMMAND release manifest", payload["provenance"]["manifest_relationship"])

    def test_duplicate_split_rows_keep_distinct_bag_ids(self):
        from pyclad.video import CommandUcfCrimeDataset

        dataset = CommandUcfCrimeDataset(
            self.root,
            concept_order=("Abuse", "Arrest"),
        )
        concepts = dataset.training_concepts()
        normal_bag_ids = [
            float(np.unique(concept.strategy_targets["bag_id"][concept.strategy_targets["weak_label"] == 0])[0])
            for concept in concepts
        ]

        self.assertEqual(len(set(normal_bag_ids)), 2)

    def test_audit_distinguishes_official_videos_from_command_manifest_rows(self):
        from pyclad.video.ucf_crime import CommandUcfCrimeAudit

        audit = CommandUcfCrimeAudit(
            root=str(self.root),
            modality="two",
            train_normal_records=810,
            train_anomaly_records=810,
            train_unique_normal_videos=800,
            train_unique_anomaly_videos=810,
            train_unique_videos=1_610,
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

        self.assertTrue(audit.official_split_counts_match)
        self.assertTrue(audit.command_release_counts_match)
        self.assertEqual(audit.train_records, 1_620)
        self.assertEqual(audit.train_unique_videos, 1_610)

    def test_cli_makes_focused_workflow_primary(self):
        from pyclad.video.cli import _parser

        audit = _parser().parse_args(["ucf-audit", "--data-root", str(self.root)])
        command = _parser().parse_args(["ucf-command", "--data-root", str(self.root)])

        self.assertFalse(audit.check_feature_arrays)
        self.assertEqual(audit.modality, "two")
        self.assertEqual(command.tasks, "T1,T2,T3")
        self.assertEqual(command.buffer_size, 1_000)
        self.assertEqual(command.replay_batch_size, 16)

        help_text = _parser().format_help()
        self.assertIn("ucf-audit", help_text)
        self.assertIn("ucf-command", help_text)
        self.assertNotIn("command-xd", help_text)
        self.assertNotIn("command-nola", help_text)
        self.assertNotIn("prism-cl-ucf", help_text)


if __name__ == "__main__":
    unittest.main()
