"""Frame evaluation and checkpoint reporting through standard callbacks."""

from pathlib import Path

from pyclad.callbacks.callback import Callback
from pyclad.output.output_writer import InfoProvider
from pyclad.video.metrics.frame import (
    compute_video_frame_metrics,
    window_scores_to_frame_scores,
)
from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy


class VideoFrameEvaluationCallback(Callback, InfoProvider):
    """Collect COMMAND task results from core scenario lifecycle events."""

    def __init__(self, strategy: ContTrainPlusPlusStrategy, checkpoint_root: Path | None = None):
        self.strategy = strategy
        self.checkpoint_root = checkpoint_root
        self.task_results = []

    def before_scenario(self, *args, **kwargs):
        """Clear reporting state when a scenario starts."""
        self.task_results = []

    def after_training(self, learned_concept, *args, **kwargs):
        """Record the completed task and optionally save its checkpoint."""
        checkpoint = None
        if self.checkpoint_root is not None:
            path = self.checkpoint_root.expanduser().resolve() / f"command-recreation-{learned_concept.name}.pt"
            self.strategy.trainer.save_checkpoint(path)
            checkpoint = str(path)
        self.task_results.append(
            {
                "task": learned_concept.name,
                "training": self.strategy.last_training,
                "evaluation": {},
                "checkpoint": checkpoint,
            }
        )

    def after_evaluation(self, evaluated_concept, window_scores, classifier_scores, windows, *args, **kwargs):
        """Evaluate temporal scores against frame labels kept on the concept."""
        labels = evaluated_concept.frame_labels
        counts = {video_id: len(values) for video_id, values in labels.items()}
        result = {}
        for name, scores in (("ddm", window_scores), ("classifier_diagnostic", classifier_scores)):
            frames = window_scores_to_frame_scores(windows, scores, counts)
            result[name] = compute_video_frame_metrics(frames, labels).as_dict()
        self.task_results[-1]["evaluation"][evaluated_concept.name] = result

    def name(self) -> str:
        return "video_frame_evaluation"

    def info(self) -> dict:
        """Return results suitable for pyCLAD output writers."""
        return {self.name(): {"task_results": self.task_results}}
