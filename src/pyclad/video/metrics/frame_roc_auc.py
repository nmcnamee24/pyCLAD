"""ROC-AUC over frame-level anomaly scores."""

import numpy as np
from sklearn.metrics import roc_auc_score

from pyclad.metrics.base.base_metric import BaseMetric


class FrameRocAuc(BaseMetric):
    """Compute frame ROC-AUC; return NaN when the ranking is undefined."""

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        """Evaluate aligned, flattened frame curves."""
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true=y_true, y_score=anomaly_scores))

    def name(self) -> str:
        return "Frame-ROC-AUC"
