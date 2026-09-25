"""ContTrain++ integration with the standard concept-incremental lifecycle."""

import numpy as np

from pyclad.strategies.strategy import ConceptIncrementalStrategy
from pyclad.video.models.command.paper_training import (
    ContTrainPlusPlusTrainer,
    PaperVideoBag,
)
from pyclad.video.prediction_results import VideoPredictionResults


class ContTrainPlusPlusStrategy(ConceptIncrementalStrategy):
    """Train and replay complete bags using COMMAND's dedicated optimizer.

    The core scenario owns iteration and callbacks. The trainer retains the
    paper-specific losses, memory updates, calibration, and replay semantics.
    Row-based TorchRunner strategies remain separate experimental baselines.
    """

    def __init__(self, trainer: ContTrainPlusPlusTrainer):
        self.trainer = trainer
        self.last_training = None

    def learn(self, data: np.ndarray) -> None:
        """Learn one task without splitting its bags into shuffled windows."""
        bags = self._bags(data)
        task_ids = {bag.task_id for bag in bags}
        if len(task_ids) != 1:
            raise ValueError("training data must contain bags from exactly one task")
        self.last_training = self.trainer.fit_task(bags, task_id=next(iter(task_ids)))

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        """Return bag-level scores and aligned temporal scores."""
        bags = self._bags(data)
        prediction = self.trainer.predict_bags(bags)
        scores = prediction["anomaly_scores"]
        bag_scores = scores.max(axis=1) if len(bags) else np.empty(0)
        return VideoPredictionResults(
            y_pred=(bag_scores >= 0.5).astype(np.int64),
            anomaly_scores=bag_scores,
            window_scores=scores.reshape(-1),
            classifier_scores=prediction["classifier_scores"].reshape(-1),
            windows=tuple(window for bag in bags for window in bag.windows),
        )

    def name(self) -> str:
        return "ContTrain++"

    def additional_info(self) -> dict:
        return self.trainer.metadata()

    @staticmethod
    def _bags(data: np.ndarray) -> tuple[PaperVideoBag, ...]:
        values = np.asarray(data, dtype=object)
        if values.ndim != 1 or any(not isinstance(bag, PaperVideoBag) for bag in values):
            raise ValueError("ContTrain++ requires a one-dimensional array of complete PaperVideoBag objects")
        return tuple(values)
