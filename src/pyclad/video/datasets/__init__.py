"""Dataset adapters provided by the video modality."""

from pyclad.video.datasets.command_ucf_crime import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    CommandUcfCrimeDataset,
    CommandUcfCrimeRecord,
)
from pyclad.video.datasets.nola import (
    NOLA_RELEVANT_CLASSES,
    NOLA_STAGE_ORDER,
    NolaContinualDataset,
    NolaGroundTruth,
    NolaPreparedTestDataset,
    extract_nola_video_features,
    load_nola_ground_truth,
)
from pyclad.video.datasets.nola_paper import (
    DARKNET_COCO_CLASSES,
    NOLA_PAPER_FEATURE_DIM,
    NolaPaperContinualDataset,
    NolaPaperPreparedTestDataset,
    build_nola_paper_trajectory_training_data,
    extract_nola_paper_video_features,
)
from pyclad.video.datasets.ucf_crime import (
    UCF_CRIME_I3D_FRAME_STEP,
    UcfCrimeI3DTestDataset,
    UcfCrimeSubsetDataset,
    load_ucf_crime_i3d_test_split,
    load_ucf_crime_window_manifest,
)

__all__ = [
    "COMMAND_UCF_CRIME_CONCEPT_ORDER",
    "CommandUcfCrimeDataset",
    "CommandUcfCrimeRecord",
    "DARKNET_COCO_CLASSES",
    "NOLA_PAPER_FEATURE_DIM",
    "NOLA_RELEVANT_CLASSES",
    "NOLA_STAGE_ORDER",
    "NolaContinualDataset",
    "NolaGroundTruth",
    "NolaPaperContinualDataset",
    "NolaPaperPreparedTestDataset",
    "NolaPreparedTestDataset",
    "UCF_CRIME_I3D_FRAME_STEP",
    "UcfCrimeI3DTestDataset",
    "UcfCrimeSubsetDataset",
    "load_ucf_crime_i3d_test_split",
    "load_ucf_crime_window_manifest",
    "extract_nola_video_features",
    "extract_nola_paper_video_features",
    "build_nola_paper_trajectory_training_data",
    "load_nola_ground_truth",
]
