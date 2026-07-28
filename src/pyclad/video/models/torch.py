"""Optional PyTorch video interface for regularization strategies."""

from __future__ import annotations

import abc

import numpy as np

from pyclad.models.model import Model
from pyclad.models.torch_backbone import TorchBackbone
from pyclad.video.prediction_results import VideoPredictionResults


class TorchVideoBackbone(TorchBackbone, Model, abc.ABC):
    """Differentiable video model accepted by existing tensor strategies.

    Implementations receive two-dimensional video-window feature tensors.
    EWC, LwF, A-GEM, and DER++ can use this interface without changes.
    """

    @abc.abstractmethod
    def fit(self, data: np.ndarray): ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> VideoPredictionResults: ...

    @abc.abstractmethod
    def name(self) -> str: ...
