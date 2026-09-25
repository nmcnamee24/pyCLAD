"""Run the UCF-Crime 4/4/5 COMMAND recreation with standard pyCLAD components."""

import argparse
from pathlib import Path

import torch

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
    """Read an existing RGB/flow archive, train, and save metrics and a checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("command-results.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dataset = CommandUcfCrimeDataset(args.data_root).read_dataset()
    model = CommandModel(config=CommandTrainerConfig(epochs=args.epochs, device=args.device, seed=args.seed))
    strategy = ContTrainPlusPlusStrategy(model)
    callbacks = [VideoFrameMetricCallback(metric) for metric in (FrameRocAuc(), FrameAveragePrecision())]
    ConceptIncrementalScenario(dataset, strategy, callbacks).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    JsonOutputWriter(args.output).write([dataset, model, strategy, *callbacks])
    model.save_checkpoint(args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
