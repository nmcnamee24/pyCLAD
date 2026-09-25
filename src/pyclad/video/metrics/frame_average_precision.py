"""Average precision over frame-level anomaly scores."""

import numpy as np
from sklearn.metrics import average_precision_score

from pyclad.metrics.base.base_metric import BaseMetric


class FrameAveragePrecision(BaseMetric):
    """Compute frame AP; return NaN for empty or single-class evaluation."""

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        """Evaluate aligned, flattened frame curves."""
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true=y_true, y_score=anomaly_scores))

    def name(self) -> str:
        return "Frame-AP"
