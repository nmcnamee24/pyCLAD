"""Model contracts that retain pyCLAD's existing array interface."""

from __future__ import annotations

import abc

import numpy as np

from pyclad.models.model import Model
from pyclad.video.prediction_results import VideoPredictionResults


class VideoAnomalyModel(Model, abc.ABC):
    """A normal pyCLAD Model operating on video-window feature matrices."""

    @abc.abstractmethod
    def fit(self, data: np.ndarray): ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> VideoPredictionResults: ...
