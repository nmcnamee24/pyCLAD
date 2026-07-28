"""Video model interfaces compatible with existing pyCLAD strategies."""

from pyclad.video.models.base import VideoAnomalyModel
from pyclad.video.models.callable import (
    CallableVideoAnomalyModel,
    CallableWeaklySupervisedVideoModel,
)
from pyclad.video.models.nola import (
    NolaFeatureLayout,
    NolaVideoModel,
    build_nola_trajectory_examples,
    nola_spatial_object_features,
    nola_temporal_object_features,
    non_maximum_suppression,
    odit_cusum,
    pack_nola_features,
)

__all__ = [
    "CallableVideoAnomalyModel",
    "CallableWeaklySupervisedVideoModel",
    "CommandVideoModel",
    "NolaFeatureLayout",
    "NolaTrajectoryPredictor",
    "NolaVideoModel",
    "TorchVideoBackbone",
    "VideoAnomalyModel",
    "build_nola_trajectory_examples",
    "nola_spatial_object_features",
    "nola_temporal_object_features",
    "non_maximum_suppression",
    "odit_cusum",
    "pack_nola_features",
]


def __getattr__(name):
    if name == "TorchVideoBackbone":
        from pyclad.video.models.torch import TorchVideoBackbone

        return TorchVideoBackbone
    if name == "CommandVideoModel":
        from pyclad.video.models.command import CommandVideoModel

        return CommandVideoModel
    if name == "NolaTrajectoryPredictor":
        from pyclad.video.models.nola import NolaTrajectoryPredictor

        return NolaTrajectoryPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
