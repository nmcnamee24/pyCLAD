"""Video anomaly detection through pyCLAD's unchanged public interfaces.

The package converts video windows into ordinary two-dimensional NumPy feature
matrices before they reach a pyCLAD strategy. Existing scenarios, strategies,
models, and replay buffers therefore require no video-specific changes.
"""

from pyclad.video.benchmarks import (
    BenchmarkResult,
    NolaBenchmarkResult,
    NolaBenchmarkRunner,
    VideoBenchmarkRunner,
)
from pyclad.video.data import (
    PrecomputedVideoDataset,
    VideoConceptsDataset,
    VideoDataset,
    VideoFeatureConcept,
    VideoStrategySchema,
    VideoWindow,
)
from pyclad.video.datasets import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    DARKNET_COCO_CLASSES,
    NOLA_PAPER_FEATURE_DIM,
    NOLA_RELEVANT_CLASSES,
    NOLA_STAGE_ORDER,
    CommandUcfCrimeDataset,
    CommandUcfCrimeRecord,
    NolaContinualDataset,
    NolaGroundTruth,
    NolaPaperContinualDataset,
    NolaPaperPreparedTestDataset,
    NolaPreparedTestDataset,
    UcfCrimeI3DTestDataset,
    UcfCrimeSubsetDataset,
    build_nola_paper_trajectory_training_data,
    extract_nola_paper_video_features,
    extract_nola_video_features,
    load_nola_ground_truth,
)
from pyclad.video.features import (
    InMemoryVideoFeatureStore,
    NpyVideoFeatureStore,
    VideoFeatureStore,
    flatten_video_curves,
    window_scores_to_frame_scores,
)
from pyclad.video.metrics import (
    AveragePrecisionDelay,
    VideoFrameMetrics,
    compute_average_precision_delay,
    compute_video_frame_metrics,
)
from pyclad.video.models import (
    CallableVideoAnomalyModel,
    CallableWeaklySupervisedVideoModel,
    DegenerateNolaScoresError,
    NolaFeatureLayout,
    NolaScoreDiagnostics,
    NolaVideoModel,
    VideoAnomalyModel,
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
from pyclad.video.prediction_results import VideoPredictionResults
from pyclad.video.preprocessing import (
    DarknetCliNolaDetector,
    DarknetNolaDetector,
    DeepSortNolaTracker,
    NolaDetection,
    SimpleIouTracker,
    TorchvisionNolaDetector,
    preprocess_nola_video,
)

__all__ = [
    "AveragePrecisionDelay",
    "BenchmarkResult",
    "CallableVideoAnomalyModel",
    "CallableWeaklySupervisedVideoModel",
    "COMMAND_UCF_CRIME_CONCEPT_ORDER",
    "DARKNET_COCO_CLASSES",
    "CommandUcfCrimeDataset",
    "CommandUcfCrimeRecord",
    "CommandVideoModel",
    "DegenerateNolaScoresError",
    "DarknetCliNolaDetector",
    "DarknetNolaDetector",
    "DeepSortNolaTracker",
    "InMemoryVideoFeatureStore",
    "NolaFeatureLayout",
    "NolaScoreDiagnostics",
    "NolaBenchmarkResult",
    "NolaBenchmarkRunner",
    "NolaContinualDataset",
    "NolaDetection",
    "NolaGroundTruth",
    "NOLA_PAPER_FEATURE_DIM",
    "NolaPaperContinualDataset",
    "NolaPaperModel",
    "NolaPaperPreparedTestDataset",
    "NolaPreparedTestDataset",
    "NOLA_RELEVANT_CLASSES",
    "NOLA_STAGE_ORDER",
    "NolaTrajectoryPredictor",
    "NolaVideoModel",
    "NpyVideoFeatureStore",
    "PrecomputedVideoDataset",
    "SimpleIouTracker",
    "TorchVideoBackbone",
    "TorchvisionNolaDetector",
    "UcfCrimeI3DTestDataset",
    "UcfCrimeSubsetDataset",
    "VideoAnomalyModel",
    "VideoBenchmarkRunner",
    "VideoConceptsDataset",
    "VideoDataset",
    "VideoFeatureConcept",
    "VideoFeatureStore",
    "VideoFrameMetrics",
    "VideoPredictionResults",
    "VideoStrategySchema",
    "VideoWindow",
    "build_nola_trajectory_examples",
    "build_nola_paper_trajectory_training_data",
    "canonical_nola_object_name",
    "compute_average_precision_delay",
    "compute_video_frame_metrics",
    "flatten_video_curves",
    "extract_nola_video_features",
    "extract_nola_paper_video_features",
    "load_nola_ground_truth",
    "nola_score_diagnostics",
    "nola_spatial_object_features",
    "nola_temporal_object_features",
    "non_maximum_suppression",
    "odit_cusum",
    "pack_nola_features",
    "preprocess_nola_video",
    "require_non_degenerate_nola_scores",
    "window_scores_to_frame_scores",
]


def __getattr__(name):
    if name in {"CommandVideoModel", "NolaPaperModel", "NolaTrajectoryPredictor", "TorchVideoBackbone"}:
        from pyclad.video import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
