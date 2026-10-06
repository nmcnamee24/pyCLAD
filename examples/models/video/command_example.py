"""Run the UCF-Crime 4/4/5 COMMAND recreation with standard pyCLAD components."""

import argparse
import logging
from pathlib import Path

from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.video.callbacks.video_frame_metric_callback import VideoFrameMetricCallback
from pyclad.video.data.command_ucf_crime import CommandUcfCrimeDataset
from pyclad.video.metrics.frame_average_precision import (
    FrameAveragePrecision,
)
from pyclad.video.metrics.frame_roc_auc import FrameRocAuc
from pyclad.video.models.command.command import CommandModel
from pyclad.video.models.command.config import CommandTrainerConfig
from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy


def main():
    """Load cached/downloaded or local RGB/flow features, train, and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, nargs="?", help="Local archive; omit to download from Hugging Face")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    parser.add_argument("--revision", help="Dataset commit, tag, or branch (default: main)")
    parser.add_argument("--local-files-only", action="store_true", help="Use an already cached dataset without network")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("command-results.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = CommandUcfCrimeDataset(
        args.data_root, cache_dir=args.cache_dir, revision=args.revision, local_files_only=args.local_files_only
    )
    model = CommandModel(config=CommandTrainerConfig(epochs=args.epochs, device=args.device, seed=args.seed))
    strategy = ContTrainPlusPlusStrategy(model)
    callbacks = [VideoFrameMetricCallback(metric) for metric in (FrameRocAuc(), FrameAveragePrecision())]
    ConceptIncrementalScenario(dataset, strategy, callbacks).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    JsonOutputWriter(args.output).write([dataset, model, strategy, *callbacks])
    model.save_checkpoint(args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
