"""Composite-loss and complete-bag ContTrain++ tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class PaperCommandTrainingTest(unittest.TestCase):
    @staticmethod
    def _architecture():
        from pyclad.video.models.command import PaperCommandArchitectureConfig

        return PaperCommandArchitectureConfig(
            appearance_dim=4,
            motion_dim=4,
            hidden_dim=8,
            state_dim=3,
            temporal_kernel_size=5,
            mamba_blocks=1,
            memory_size=5,
            projection_dim=6,
            cbam_reduction=2,
            cbam_temporal_kernel_size=7,
            dropout=0.0,
        )

    def test_composite_loss_is_finite_and_zeros_positive_dependent_terms_without_positives(self):
        from pyclad.video.models.command import (
            PaperCommandNetwork,
            paper_command_composite_loss,
        )

        model = PaperCommandNetwork(self._architecture())
        appearance = torch.randn(4, 8, 4)
        motion = torch.randn(4, 8, 4)
        mixed = paper_command_composite_loss(model(appearance, motion), torch.tensor([0.0, 0.0, 1.0, 1.0]))
        normal_only = paper_command_composite_loss(model(appearance, motion), torch.zeros(4))

        self.assertTrue(torch.isfinite(mixed.total))
        self.assertGreaterEqual(float(mixed.mil.detach()), 0.0)
        self.assertGreaterEqual(float(mixed.contrastive.detach()), 0.0)
        self.assertEqual(float(normal_only.contrastive.detach()), 0.0)
        self.assertEqual(float(normal_only.anomaly_separation.detach()), 0.0)

    def test_replay_selection_balances_labels_tasks_and_keeps_complete_bags(self):
        from pyclad.video.models.command import (
            ContTrainReplayBuffer,
            PaperVideoBag,
            ReplayEntry,
        )

        buffer = ContTrainReplayBuffer(max_size=8)
        entries = []
        for index in range(12):
            bag = PaperVideoBag(
                bag_id=f"bag-{index:02d}",
                task_id=f"T{index % 3 + 1}",
                features=np.full((8, 8), index, dtype=np.float32),
                weak_label=index % 2,
            )
            curve = np.zeros(8, dtype=np.float32)
            curve[index % 8] = index + 1
            entries.append(ReplayEntry(bag, curve))
        buffer.add(entries)
        sample = buffer.sample(6)

        self.assertEqual(len(buffer), 8)
        self.assertEqual(len(sample), 6)
        self.assertEqual(sum(bag.weak_label == 0 for bag in sample), 3)
        self.assertEqual(sum(bag.weak_label == 1 for bag in sample), 3)
        self.assertTrue(all(bag.features.shape == (8, 8) for bag in sample))
        self.assertGreaterEqual(len({bag.task_id for bag in sample}), 2)

    def test_optimizer_uses_slow_secondary_memory_group_and_checkpoint_resumes(self):
        from pyclad.video.models.command import (
            ContTrainPlusPlusTrainer,
            PaperCommandTrainerConfig,
            PaperCommandVideoModel,
            PaperVideoBag,
        )

        config = PaperCommandTrainerConfig(
            architecture=self._architecture(),
            epochs=1,
            batch_size=4,
            replay_batch_size=2,
            buffer_size=8,
            learning_rate=1e-3,
            secondary_memory_learning_rate=1e-4,
            novelty_warmup_epochs=10,
            device="cpu",
        )
        trainer = ContTrainPlusPlusTrainer(PaperCommandVideoModel(config.architecture), config)
        bags = tuple(
            PaperVideoBag(
                bag_id=f"video-{index}",
                task_id="T1",
                features=np.random.default_rng(index).normal(size=(8, 8)).astype(np.float32),
                weak_label=index % 2,
            )
            for index in range(4)
        )
        result = trainer.fit_task(bags, task_id="T1")
        before = trainer.predict_bags(bags, batch_size=2)["raw_scores"]

        self.assertEqual([group["name"] for group in trainer.optimizer.param_groups], ["primary", "secondary_memory"])
        self.assertEqual(result["buffer_bags"], 4)
        self.assertTrue(np.isfinite(before).all())

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint)
            restored = ContTrainPlusPlusTrainer(PaperCommandVideoModel(config.architecture), config)
            restored.load_checkpoint(checkpoint)
            after = restored.predict_bags(bags, batch_size=2)["raw_scores"]

        np.testing.assert_array_equal(before, after)
        self.assertEqual(len(restored.replay), 4)

    def test_calibration_is_bounded_monotonic_and_retains_extreme_tail_rank(self):
        from pyclad.video.models.command import (
            ContTrainPlusPlusTrainer,
            PaperCommandTrainerConfig,
            PaperCommandVideoModel,
            PaperVideoBag,
        )

        config = PaperCommandTrainerConfig(
            architecture=self._architecture(),
            epochs=1,
            batch_size=2,
            replay_batch_size=1,
            device="cpu",
        )
        trainer = ContTrainPlusPlusTrainer(
            PaperCommandVideoModel(config.architecture, mode="normal-only"),
            config,
        )
        bags = tuple(
            PaperVideoBag(
                bag_id=f"tail-{index}",
                task_id="test",
                features=np.full((8, 8), -1000.0 + index, dtype=np.float32),
                weak_label=0,
            )
            for index in range(2)
        )
        trainer._calibration_median = 1000.0
        trainer._calibration_scale = 0.01
        scores = trainer.predict_bags(bags)["anomaly_scores"]

        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))
        self.assertGreater(len(np.unique(scores)), 1)

    def test_learning_rate_schedule_restarts_at_each_task_boundary(self):
        from pyclad.video.models.command import (
            ContTrainPlusPlusTrainer,
            PaperCommandTrainerConfig,
            PaperCommandVideoModel,
            PaperVideoBag,
        )

        config = PaperCommandTrainerConfig(
            architecture=self._architecture(),
            epochs=2,
            batch_size=2,
            replay_batch_size=1,
            buffer_size=8,
            learning_rate=1e-3,
            secondary_memory_learning_rate=1e-4,
            lr_step_size=1,
            lr_gamma=0.1,
            novelty_warmup_epochs=10,
            device="cpu",
        )
        trainer = ContTrainPlusPlusTrainer(PaperCommandVideoModel(config.architecture), config)

        def bags(task):
            return tuple(
                PaperVideoBag(
                    bag_id=f"{task}-{index}",
                    task_id=task,
                    features=np.random.default_rng(index).normal(size=(8, 8)).astype(np.float32),
                    weak_label=index % 2,
                )
                for index in range(2)
            )

        first = trainer.fit_task(bags("T1"), task_id="T1")
        second = trainer.fit_task(bags("T2"), task_id="T2")

        self.assertAlmostEqual(first["learning_rates"]["primary"], 1e-5)
        self.assertAlmostEqual(second["learning_rates"]["primary"], 1e-5)
        self.assertAlmostEqual(second["learning_rates"]["secondary_memory"], 1e-6)

    def test_bag_adapter_rejects_incompatible_1024_dimensional_cache(self):
        from pyclad.video.data import VideoFeatureConcept, VideoWindow
        from pyclad.video.models.command import bags_from_concept

        concept = VideoFeatureConcept(
            name="old-cache",
            features=np.zeros((32, 1024), dtype=np.float32),
            windows=tuple(
                VideoWindow("video", index, index, index, label=0, payload={"window_index": index})
                for index in range(32)
            ),
        )
        with self.assertRaisesRegex(ValueError, "requires 2048-D"):
            bags_from_concept(concept)

    def test_duplicate_archive_paths_remain_distinct_record_bags(self):
        from pyclad.video.data import VideoFeatureConcept, VideoWindow
        from pyclad.video.models.command import bags_from_concept

        windows = []
        for record in ("normal:0001", "normal:0002"):
            for index in range(32):
                windows.append(
                    VideoWindow(
                        "Normal_Videos_event/duplicate.mp4",
                        index,
                        index,
                        len(windows),
                        label=0,
                        payload={"window_index": index, "record_instance_id": record},
                    )
                )
        concept = VideoFeatureConcept(
            name="T3",
            features=np.zeros((64, 2048), dtype=np.float32),
            windows=tuple(windows),
        )
        bags = bags_from_concept(concept, task_id="T3")

        self.assertEqual([bag.bag_id for bag in bags], ["normal:0001", "normal:0002"])
        self.assertTrue(all(bag.features.shape == (32, 2048) for bag in bags))


if __name__ == "__main__":
    unittest.main()
