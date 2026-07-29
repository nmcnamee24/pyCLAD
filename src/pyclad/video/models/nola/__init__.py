"""Modern pyCLAD adaptation of NOLA."""

from pyclad.video.models.nola.features import (
    NolaFeatureLayout,
    build_nola_trajectory_examples,
    canonical_nola_object_name,
    nola_spatial_object_features,
    nola_temporal_object_features,
    pack_nola_features,
)
from pyclad.video.models.nola.model import NolaVideoModel
from pyclad.video.models.nola.scoring import (
    DegenerateNolaScoresError,
    NolaScoreDiagnostics,
    nola_score_diagnostics,
    non_maximum_suppression,
    odit_cusum,
    require_non_degenerate_nola_scores,
)

__all__ = [
    "DegenerateNolaScoresError",
    "NolaFeatureLayout",
    "NolaPaperModel",
    "NolaScoreDiagnostics",
    "NolaTrajectoryPredictor",
    "NolaVideoModel",
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
    if name == "NolaPaperModel":
        from pyclad.video.models.nola.paper import NolaPaperModel

        return NolaPaperModel
    if name == "NolaTrajectoryPredictor":
        from pyclad.video.models.nola.trajectory import NolaTrajectoryPredictor

        return NolaTrajectoryPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
