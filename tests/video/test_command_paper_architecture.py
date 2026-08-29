"""Structural tests for the paper-aligned COMMAND architecture."""

from __future__ import annotations

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "COMMAND requires the optional torch dependency")
class PaperCommandArchitectureTest(unittest.TestCase):
    def _small_config(self):
        from pyclad.video.models.command.paper_architecture import (
            PaperCommandArchitectureConfig,
        )

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

    def test_published_defaults_are_fixed_in_config(self):
        from pyclad.video.models.command.paper_architecture import (
            PaperCommandArchitectureConfig,
        )

        config = PaperCommandArchitectureConfig()

        self.assertEqual(config.appearance_dim, 1024)
        self.assertEqual(config.motion_dim, 1024)
        self.assertEqual(config.fused_dim, 2048)
        self.assertEqual(config.temporal_kernel_size, 5)
        self.assertEqual(config.memory_size, 256)

    def test_default_parameter_count_is_locked(self):
        from pyclad.video.models.command.paper_architecture import PaperCommandNetwork

        model = PaperCommandNetwork()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 15_036_943)

    def test_complete_sequence_reaches_every_paper_module(self):
        import torch

        from pyclad.video.models.command.paper_architecture import PaperCommandNetwork

        config = self._small_config()
        model = PaperCommandNetwork(config).eval()
        appearance = torch.randn(2, 32, config.appearance_dim)
        motion = torch.randn(2, 32, config.motion_dim)

        with torch.no_grad():
            output = model(appearance, motion)

        self.assertEqual(output.logits.shape, (2, 32))
        self.assertEqual(output.probabilities.shape, (2, 32))
        self.assertEqual(output.projections.shape, (2, 32, config.projection_dim))
        self.assertEqual(output.temporal_features.shape, (2, 32, config.fused_dim))
        self.assertEqual(output.memory.primary_distances.shape, (2, 32, config.memory_size))
        self.assertEqual(output.memory.secondary_distances.shape, (2, 32, config.memory_size))
        self.assertEqual(output.memory.dual_memory_deviation.shape, (2, 32))
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_augfusenet_is_exact_modality_concatenation(self):
        import torch

        from pyclad.video.models.command.paper_architecture import PaperAugFuseNet

        appearance = torch.tensor([[[1.0, 2.0]]])
        motion = torch.tensor([[[3.0, 4.0, 5.0]]])

        fused = PaperAugFuseNet(2, 3)(appearance, motion)

        torch.testing.assert_close(fused, torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]))

    def test_temporal_convolution_has_published_kernel_and_is_depthwise(self):
        from pyclad.video.models.command.paper_architecture import PaperCommandNetwork

        config = self._small_config()
        model = PaperCommandNetwork(config)
        convolution = model.temporal.blocks[0].depthwise_convolution

        self.assertEqual(convolution.kernel_size, (5,))
        self.assertEqual(convolution.groups, config.hidden_dim)
        self.assertEqual(model.temporal.cbam.output_norm.normalized_shape[0], config.fused_dim)

    def test_memdualnet_uses_exact_euclidean_minimum(self):
        import torch

        from pyclad.video.models.command.paper_architecture import PaperMemDualNet

        memory = PaperMemDualNet(feature_dim=2, memory_size=2)
        with torch.no_grad():
            memory.primary_memory.copy_(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
            memory.secondary_memory.copy_(torch.tensor([[2.0, 0.0], [20.0, 20.0]]))

        output = memory(torch.tensor([[[1.0, 0.0]]]))

        torch.testing.assert_close(output.primary_nearest_distance, torch.tensor([[1.0]]))
        torch.testing.assert_close(output.secondary_nearest_distance, torch.tensor([[1.0]]))
        torch.testing.assert_close(output.dual_memory_deviation, torch.tensor([[1.0]]))

    def test_frequency_aware_fifo_lru_replacement_is_deterministic(self):
        import torch

        from pyclad.video.models.command.paper_architecture import PaperMemDualNet

        memory = PaperMemDualNet(feature_dim=2, memory_size=3)
        memory.primary_access_count.copy_(torch.tensor([2, 0, 0]))
        memory.primary_last_access.copy_(torch.tensor([3, 2, 1]))

        replaced = memory.replace_least_accessed("primary", torch.tensor([7.0, 8.0]))

        self.assertEqual(replaced, 2)
        torch.testing.assert_close(memory.primary_memory[2], torch.tensor([7.0, 8.0]))


if __name__ == "__main__":
    unittest.main()
