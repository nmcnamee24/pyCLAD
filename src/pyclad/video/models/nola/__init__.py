"""Modern pyCLAD adaptation of NOLA."""

from pyclad.video.models.nola.features import (
    NolaFeatureLayout,
    build_nola_trajectory_examples,
    nola_spatial_object_features,
    nola_temporal_object_features,
    pack_nola_features,
)
from pyclad.video.models.nola.model import NolaVideoModel
from pyclad.video.models.nola.scoring import (
    non_maximum_suppression,
    odit_cusum,
)

__all__ = [
    "NolaFeatureLayout",
    "NolaTrajectoryPredictor",
    "NolaVideoModel",
    "build_nola_trajectory_examples",
    "nola_spatial_object_features",
    "nola_temporal_object_features",
    "non_maximum_suppression",
    "odit_cusum",
    "pack_nola_features",
]


def __getattr__(name):
    if name == "NolaTrajectoryPredictor":
        from pyclad.video.models.nola.trajectory import NolaTrajectoryPredictor

        return NolaTrajectoryPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
