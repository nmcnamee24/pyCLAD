"""Prediction results with optional video-level projections."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from pyclad.output.prediction_results import PredictionResults


@dataclass
class VideoPredictionResults(PredictionResults):
    predicted_classes: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    window_scores: Optional[np.ndarray] = None
    frame_scores: Optional[Dict[str, np.ndarray]] = None
