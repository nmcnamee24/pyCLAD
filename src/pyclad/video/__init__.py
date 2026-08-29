"""Video anomaly detection centered on UCF-Crime and COMMAND.

Dataset, scenario, audit, and metric contracts stay importable without
PyTorch. Model and trainer symbols are loaded lazily when the optional
``video`` dependency is installed.
"""

from pyclad.video.benchmarks import BenchmarkResult, VideoBenchmarkRunner
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
    COMMAND_UCF_CRIME_PAPER_TASKS,
    CommandUcfCrimeDataset,
    CommandUcfCrimeRecord,
    UcfCrimeI3DTestDataset,
    UcfCrimeSubsetDataset,
)
from pyclad.video.features import (
    InMemoryVideoFeatureStore,
    NpyVideoFeatureStore,
    VideoFeatureStore,
    flatten_video_curves,
    window_scores_to_frame_scores,
)
from pyclad.video.metrics import VideoFrameMetrics, compute_video_frame_metrics
from pyclad.video.models import (
    CallableVideoAnomalyModel,
    CallableWeaklySupervisedVideoModel,
    VideoAnomalyModel,
)
from pyclad.video.prediction_results import VideoPredictionResults
from pyclad.video.ucf_crime import (
    CLASSWISE_PROTOCOL,
    COMMAND_PAPER_RECREATION_PROTOCOL,
    COMMAND_UCF_CRIME_PROTOCOLS,
    HELD_OUT_LAST_PROTOCOL,
    PAPER_PROTOCOL,
    CommandUcfCrimeAudit,
    CommandUcfCrimeScenario,
    CommandUcfCrimeTask,
    audit_command_ucf_crime,
    build_command_ucf_crime_scenario,
)

__all__ = [
    "BenchmarkResult",
    "CallableVideoAnomalyModel",
    "CallableWeaklySupervisedVideoModel",
    "CLASSWISE_PROTOCOL",
    "COMMAND_PAPER_RECREATION_PROTOCOL",
    "COMMAND_UCF_CRIME_CONCEPT_ORDER",
    "COMMAND_UCF_CRIME_PAPER_TASKS",
    "COMMAND_UCF_CRIME_PROTOCOLS",
    "CommandNormalOnlyModel",
    "CommandUcfCrimeAudit",
    "CommandUcfCrimeDataset",
    "CommandUcfCrimeRecord",
    "CommandUcfCrimeScenario",
    "CommandUcfCrimeTask",
    "CommandVideoModel",
    "ContTrainPlusPlusTrainer",
    "HELD_OUT_LAST_PROTOCOL",
    "InMemoryVideoFeatureStore",
    "NpyVideoFeatureStore",
    "PrecomputedVideoDataset",
    "PAPER_PROTOCOL",
    "PaperCommandTrainerConfig",
    "PaperCommandVideoModel",
    "TorchVideoBackbone",
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
    "compute_video_frame_metrics",
    "audit_command_ucf_crime",
    "build_command_ucf_crime_scenario",
    "flatten_video_curves",
    "window_scores_to_frame_scores",
]


def __getattr__(name):
    if name in {
        "CommandNormalOnlyModel",
        "CommandVideoModel",
        "ContTrainPlusPlusTrainer",
        "PaperCommandTrainerConfig",
        "PaperCommandVideoModel",
        "TorchVideoBackbone",
    }:
        from pyclad.video.models import TorchVideoBackbone

        if name == "TorchVideoBackbone":
            value = TorchVideoBackbone
        else:
            from pyclad.video.models import command

            value = getattr(command, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
