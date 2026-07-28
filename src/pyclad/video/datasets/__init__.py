"""Dataset adapters provided by the video modality."""

from pyclad.video.datasets.ucf_crime import (
    UCF_CRIME_I3D_FRAME_STEP,
    UcfCrimeI3DTestDataset,
    UcfCrimeSubsetDataset,
    load_ucf_crime_i3d_test_split,
    load_ucf_crime_window_manifest,
)

__all__ = [
    "UCF_CRIME_I3D_FRAME_STEP",
    "UcfCrimeI3DTestDataset",
    "UcfCrimeSubsetDataset",
    "load_ucf_crime_i3d_test_split",
    "load_ucf_crime_window_manifest",
]
