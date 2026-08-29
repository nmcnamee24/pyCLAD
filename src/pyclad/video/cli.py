"""Command-line workflows for UCF-Crime and the COMMAND recreation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected video workflow."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    _set_global_seed(arguments.seed)
    if arguments.command == "ucf-audit":
        _run_ucf_audit(arguments)
    elif arguments.command in {"ucf-command", "command-paper"}:
        _run_command_paper(arguments)
    else:  # pragma: no cover - argparse enforces the available choices
        parser.error(f"unknown command: {arguments.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pyCLAD video anomaly detection, centered on UCF-Crime and COMMAND")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{ucf-audit,ucf-command}",
    )

    audit = commands.add_parser(
        "ucf-audit",
        help="audit a COMMAND-formatted UCF-Crime feature archive",
    )
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--modality", choices=("two", "rgb", "flow"), default="two")
    audit.add_argument(
        "--check-feature-arrays",
        action="store_true",
        help="open every feature array and validate its shape",
    )
    _add_reproducibility_arguments(audit)

    command = commands.add_parser(
        "ucf-command",
        help="run the full-bag recreation of COMMAND's UCF-Crime 4-4-5 workflow",
    )
    _add_command_arguments(command)

    legacy = commands.add_parser("command-paper", help=argparse.SUPPRESS)
    _add_command_arguments(legacy)

    # Keep the historical alias callable without advertising it as a second
    # supported workflow in top-level help.
    commands._choices_actions = [
        choice for choice in commands._choices_actions if choice.dest in {"ucf-audit", "ucf-command"}
    ]
    return parser


def _add_command_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tasks", default="T1,T2,T3")
    parser.add_argument("--videos-per-class", type=_nonnegative_int, default=0)
    parser.add_argument("--test-normal-videos", type=_nonnegative_int, default=0)
    parser.add_argument("--test-anomaly-videos", type=_nonnegative_int, default=0)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-batch-size", type=_nonnegative_int, default=16)
    parser.add_argument("--buffer-size", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--secondary-memory-learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.15)
    parser.add_argument("--mil-margin", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.5)
    parser.add_argument("--anomaly-margin", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=8e-5)
    parser.add_argument("--smoothness-weight", type=float, default=8e-5)
    parser.add_argument("--score-l2-weight", type=float, default=5e-5)
    parser.add_argument("--device", default=_default_torch_device())
    _add_reproducibility_arguments(parser)


def _run_ucf_audit(arguments: argparse.Namespace) -> None:
    from pyclad.video.ucf_crime import audit_command_ucf_crime

    report = audit_command_ucf_crime(
        arguments.data_root,
        modality=arguments.modality,
        check_feature_arrays=arguments.check_feature_arrays,
    )
    _emit_json(
        {
            "method": "COMMAND-UCF-Crime-archive-audit",
            "claim": "read-only package preflight; no model training performed",
            "audit": report.as_dict(),
        },
        arguments,
    )


def _paper_trainer_config(arguments: argparse.Namespace):
    from pyclad.video.models.command import (
        PaperCommandArchitectureConfig,
        PaperCommandLossConfig,
        PaperCommandTrainerConfig,
    )

    return PaperCommandTrainerConfig(
        architecture=PaperCommandArchitectureConfig(),
        loss=PaperCommandLossConfig(
            contrastive_temperature=arguments.contrastive_temperature,
            mil_margin=arguments.mil_margin,
            focal_alpha=arguments.focal_alpha,
            focal_gamma=arguments.focal_gamma,
            anomaly_margin=arguments.anomaly_margin,
            sparsity_weight=arguments.sparsity_weight,
            smoothness_weight=arguments.smoothness_weight,
            score_l2_weight=arguments.score_l2_weight,
        ),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        replay_batch_size=arguments.replay_batch_size,
        buffer_size=arguments.buffer_size,
        learning_rate=arguments.learning_rate,
        secondary_memory_learning_rate=arguments.secondary_memory_learning_rate,
        gradient_clip=arguments.gradient_clip,
        lr_step_size=arguments.lr_step_size,
        lr_gamma=arguments.lr_gamma,
        seed=arguments.seed,
        device=arguments.device,
    )


def _run_command_paper(arguments: argparse.Namespace) -> None:
    from pyclad.video import (
        CommandUcfCrimeDataset,
        audit_command_ucf_crime,
        build_command_ucf_crime_scenario,
    )
    from pyclad.video.models.command import (
        ContTrainPlusPlusTrainer,
        PaperCommandVideoModel,
        bags_from_concept,
    )

    archive_audit = audit_command_ucf_crime(arguments.data_root, check_feature_arrays=True)
    if not archive_audit.ready:
        raise ValueError("COMMAND UCF-Crime archive preflight failed; run ucf-audit for details")

    dataset = CommandUcfCrimeDataset(arguments.data_root)
    available = dataset.paper_training_tasks(max_videos_per_class=_limit(arguments.videos_per_class))
    selected_names = _csv(arguments.tasks)
    unknown = sorted(set(selected_names) - {task.name for task in available})
    if unknown:
        raise ValueError(f"unknown COMMAND recreation tasks: {unknown}")
    tasks = tuple(task for task in available if task.name in selected_names)

    config = _paper_trainer_config(arguments)
    trainer = ContTrainPlusPlusTrainer(PaperCommandVideoModel(config.architecture), config)
    test = dataset.test_concept(
        max_normal_videos=_limit(arguments.test_normal_videos),
        max_anomaly_videos=_limit(arguments.test_anomaly_videos),
    )
    task_results = []
    for task in tasks:
        print(f"[COMMAND-RECREATION] training {task.name}", file=sys.stderr, flush=True)
        task_bags = bags_from_concept(task, task_id=task.name)
        training = trainer.fit_task(task_bags, task_id=task.name)
        evaluation = _evaluate_paper_predictions(dataset, test, trainer)
        checkpoint = None
        if arguments.checkpoint_root is not None:
            checkpoint_path = arguments.checkpoint_root.expanduser().resolve() / f"command-recreation-{task.name}.pt"
            trainer.save_checkpoint(checkpoint_path)
            checkpoint = str(checkpoint_path)
        task_results.append(
            {
                "task": task.name,
                "training": training,
                "evaluation": evaluation,
                "checkpoint": checkpoint,
            }
        )

    _emit_json(
        {
            "method": "COMMAND-paper-recreation",
            "claim": (
                "PyCLAD clean-room recreation of the COMMAND architecture and "
                "Section IV-C 4-4-5 stream; not author code or an exact numerical reproduction"
            ),
            "archive_audit": archive_audit.as_dict(),
            "scenario": build_command_ucf_crime_scenario().as_dict(),
            "protocol": {
                "continual_tasks": [task.name for task in tasks],
                "temporal_bagging": "complete 32-window video bags",
                "generic_row_strategies_used": False,
                "primary_score": "calibrated dual-memory deviation",
                "classifier_score": "diagnostic only",
            },
            "trainer": trainer.metadata(),
            "train_manifest_records": sum(len(bags_from_concept(task, task_id=task.name)) for task in tasks),
            "train_unique_video_ids": len({window.video_id for task in tasks for window in task.windows}),
            "test_videos": len({window.video_id for window in test.windows}),
            "task_results": task_results,
        },
        arguments,
    )


def _evaluate_paper_predictions(dataset, test, trainer) -> dict[str, object]:
    from pyclad.video import compute_video_frame_metrics, window_scores_to_frame_scores
    from pyclad.video.models.command import bags_from_concept

    bags = bags_from_concept(test, task_id="test")
    predictions = trainer.predict_bags(bags)
    windows = tuple(window for bag in bags for window in bag.windows)
    selected_ids = {bag.bag_id for bag in bags}
    labels = {video_id: values for video_id, values in dataset.frame_labels("test").items() if video_id in selected_ids}
    frame_counts = {video_id: len(values) for video_id, values in labels.items()}
    anomaly_frames = window_scores_to_frame_scores(
        windows,
        predictions["anomaly_scores"].reshape(-1),
        frame_counts,
    )
    classifier_frames = window_scores_to_frame_scores(
        windows,
        predictions["classifier_scores"].reshape(-1),
        frame_counts,
    )
    return {
        "ddm": compute_video_frame_metrics(anomaly_frames, labels).as_dict(),
        "classifier_diagnostic": compute_video_frame_metrics(classifier_frames, labels).as_dict(),
    }


def _default_torch_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("comma-separated argument must contain at least one value")
    return result


def _limit(value: int) -> int | None:
    return None if value == 0 else value


def _add_reproducibility_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="write the structured result to this path in addition to stdout",
    )


def _nonnegative_int(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return integer


def _set_global_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _emit_json(payload: dict, arguments: argparse.Namespace) -> None:
    record = {
        **payload,
        "run": {
            "command": arguments.command,
            "seed": arguments.seed,
            "commit_sha": _commit_sha(),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "arguments": _argument_payload(arguments),
            "runtime": _runtime_metadata(),
        },
    }
    non_finite = list(_non_finite_paths(record))
    record["validation"] = {
        "finite": not non_finite,
        "non_finite_values": non_finite,
    }
    encoded = json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n"
    if arguments.output_json is not None:
        output_path = arguments.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(encoded, encoding="utf-8")
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    print(encoded, end="")


def _argument_payload(arguments: argparse.Namespace) -> dict[str, Any]:
    return {name: str(value) if isinstance(value, Path) else value for name, value in sorted(vars(arguments).items())}


def _commit_sha() -> str | None:
    supplied = os.environ.get("PYCLAD_COMMIT_SHA")
    if supplied:
        return supplied.strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__,
    }
    try:
        import torch
    except ImportError:
        metadata["torch"] = None
        return metadata
    metadata.update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    )
    if torch.cuda.is_available():
        metadata["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return metadata


def _non_finite_paths(value: Any, path: str = "$") -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _non_finite_paths(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _non_finite_paths(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        yield path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
