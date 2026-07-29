"""Video model interfaces compatible with existing pyCLAD strategies."""

from pyclad.video.models.base import VideoAnomalyModel
from pyclad.video.models.callable import (
    CallableVideoAnomalyModel,
    CallableWeaklySupervisedVideoModel,
)
from pyclad.video.models.nola import (
    DegenerateNolaScoresError,
    NolaFeatureLayout,
    NolaScoreDiagnostics,
    NolaVideoModel,
    build_nola_trajectory_examples,
    canonical_nola_object_name,
    nola_score_diagnostics,
    nola_spatial_object_features,
    nola_temporal_object_features,
    non_maximum_suppression,
    odit_cusum,
    pack_nola_features,
    require_non_degenerate_nola_scores,
)

__all__ = [
    "CallableVideoAnomalyModel",
    "CallableWeaklySupervisedVideoModel",
    "CommandVideoModel",
    "DegenerateNolaScoresError",
    "NolaFeatureLayout",
    "NolaPaperModel",
    "NolaScoreDiagnostics",
    "NolaTrajectoryPredictor",
    "NolaVideoModel",
    "TorchVideoBackbone",
    "VideoAnomalyModel",
    "build_nola_trajectory_examples",
    "canonical_nola_object_name",
    "nola_score_diagnostics",
    "nola_spatial_object_features",
    "nola_temporal_object_features",
    "non_maximum_suppression",
    "odit_cusum",
    "pack_nola_features",
    "require_non_degenerate_nola_scores",
]


def __getattr__(name):
    if name == "TorchVideoBackbone":
        from pyclad.video.models.torch import TorchVideoBackbone

        return TorchVideoBackbone
    if name == "CommandVideoModel":
        from pyclad.video.models.command import CommandVideoModel

        return CommandVideoModel
    if name in {"NolaPaperModel", "NolaTrajectoryPredictor"}:
        from pyclad.video.models.nola import NolaTrajectoryPredictor

        if name == "NolaPaperModel":
            from pyclad.video.models.nola import NolaPaperModel

            return NolaPaperModel
        return NolaTrajectoryPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
