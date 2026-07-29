"""Runnable COMMAND and NOLA workflows for ``python -m pyclad.video``."""

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
from typing import Any, Callable, Iterator, Sequence

import numpy as np


COMMAND_STRATEGIES = (
    "naive",
    "cumulative",
    "mste",
    "replay-only",
    "replay-enhanced",
    "ewc",
    "lwf",
    "agem",
    "der++",
)
NOLA_STRATEGIES = (
    "naive",
    "cumulative",
    "mste",
    "replay-only",
    "replay-enhanced",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    _set_global_seed(arguments.seed)
    if arguments.command == "command":
        _run_command(arguments)
    elif arguments.command == "nola-preprocess":
        _preprocess_nola(arguments)
    elif arguments.command == "nola":
        _run_nola(arguments)
    else:
        parser.error(f"unknown command: {arguments.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pyCLAD video anomaly detection workflows")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("command", help="run COMMAND on its UCF-Crime RGB/flow archive")
    command.add_argument("--data-root", required=True)
    command.add_argument("--strategy", choices=COMMAND_STRATEGIES, default="cumulative")
    command.add_argument("--concepts", default="Abuse,Arrest")
    command.add_argument("--videos-per-class", type=int, default=1)
    command.add_argument("--test-normal-videos", type=int, default=1)
    command.add_argument("--test-anomaly-videos", type=int, default=1)
    command.add_argument("--epochs", type=int, default=1)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--buffer-size", type=int, default=256)
    command.add_argument("--hidden-dim", type=int, default=128)
    command.add_argument("--embedding-dim", type=int, default=128)
    command.add_argument("--memory-size", type=int, default=64)
    command.add_argument("--device", default=_default_torch_device())
    _add_reproducibility_arguments(command)

    preprocess = commands.add_parser("nola-preprocess", help="detect and track NOLA test MP4s")
    preprocess.add_argument("--data-root", required=True)
    preprocess.add_argument("--output-root", required=True)
    preprocess.add_argument("--video-ids", required=True)
    preprocess.add_argument("--frame-stride", type=int, default=1)
    preprocess.add_argument("--max-frames", type=int)
    preprocess.add_argument("--confidence-threshold", type=float, default=0.25)
    preprocess.add_argument("--device", default="cpu")
    preprocess.add_argument("--overwrite", action="store_true")
    _add_reproducibility_arguments(preprocess)

    nola = commands.add_parser("nola", help="run NOLA on staged train data and prepared test data")
    nola.add_argument("--data-root", required=True)
    nola.add_argument("--processed-test-root", required=True)
    nola.add_argument("--ground-truth", required=True)
    nola.add_argument("--strategy", choices=NOLA_STRATEGIES, default="cumulative")
    nola.add_argument("--stages", default="M-Train,Train0")
    nola.add_argument("--video-ids")
    nola.add_argument("--frame-stride", type=int, default=30)
    nola.add_argument("--videos-per-stage", type=int, default=2)
    nola.add_argument("--frames-per-video", type=int, default=120)
    nola.add_argument("--buffer-size", type=int, default=512)
    nola.add_argument("--neighbors", type=int, default=5)
    nola.add_argument("--odit", action="store_true")
    nola.add_argument("--drift", type=float, default=7.0)
    _add_reproducibility_arguments(nola)
    return parser


def _run_command(arguments: argparse.Namespace) -> None:
    from pyclad.video import (
        CommandUcfCrimeDataset,
        CommandVideoModel,
        compute_video_frame_metrics,
        window_scores_to_frame_scores,
    )

    dataset = CommandUcfCrimeDataset(arguments.data_root)
    concepts = dataset.training_concepts(
        concepts=_csv(arguments.concepts),
        max_videos_per_class=_limit(arguments.videos_per_class),
    )
    test = dataset.test_concept(
        max_normal_videos=_limit(arguments.test_normal_videos),
        max_anomaly_videos=_limit(arguments.test_anomaly_videos),
    )

    def model():
        return CommandVideoModel(
            dataset.feature_dim,
            strategy_schema=dataset.strategy_schema,
            hidden_dim=arguments.hidden_dim,
            embedding_dim=arguments.embedding_dim,
            memory_size=arguments.memory_size,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            device=arguments.device,
        )

    strategy = _command_strategy(
        arguments.strategy,
        model,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        buffer_size=arguments.buffer_size,
        device=arguments.device,
    )
    for concept in concepts:
        if arguments.strategy == "mste":
            strategy.learn(concept.strategy_matrix(), concept_id=concept.name)
        else:
            strategy.learn(concept.strategy_matrix())
    if arguments.strategy == "mste":
        prediction = strategy.predict(test.features, concept_id=concepts[-1].name)
    else:
        prediction = strategy.predict(test.features)

    selected_ids = {window.video_id for window in test.windows}
    labels = {
        video_id: values
        for video_id, values in dataset.frame_labels().items()
        if video_id in selected_ids
    }
    frame_scores = window_scores_to_frame_scores(
        test.windows,
        prediction.anomaly_scores,
        {video_id: len(values) for video_id, values in labels.items()},
    )
    metrics = compute_video_frame_metrics(frame_scores, labels)
    _emit_json(
        {
            "method": "COMMAND",
            "strategy": strategy.name(),
            "device": arguments.device,
            "train_concepts": [concept.name for concept in concepts],
            "train_videos": _training_video_records(concepts),
            "train_unique_video_ids": _unique_training_video_ids(concepts),
            "train_rows": sum(len(concept.features) for concept in concepts),
            "test_rows": len(test.features),
            "test_videos": len(selected_ids),
            "metrics": metrics.as_dict(),
            "model": {
                "hidden_dim": arguments.hidden_dim,
                "embedding_dim": arguments.embedding_dim,
                "memory_size": arguments.memory_size,
            },
        },
        arguments,
    )


def _preprocess_nola(arguments: argparse.Namespace) -> None:
    from pyclad.video import TorchvisionNolaDetector, preprocess_nola_video

    data_root = Path(arguments.data_root).expanduser().resolve()
    output_root = Path(arguments.output_root).expanduser().resolve()
    detector = TorchvisionNolaDetector(
        confidence_threshold=arguments.confidence_threshold,
        device=arguments.device,
    )
    video_ids = _csv(arguments.video_ids)
    if video_ids == ("all",):
        video_ids = tuple(
            path.name
            for path in sorted((data_root / "Test").iterdir())
            if path.is_dir()
        )
    outputs = []
    for video_id in video_ids:
        output = preprocess_nola_video(
            data_root / "Test" / video_id / "video.mp4",
            output_root / video_id,
            detector,
            frame_stride=arguments.frame_stride,
            max_frames=arguments.max_frames,
            overwrite=arguments.overwrite,
        )
        metadata_path = output / "metadata.json"
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        outputs.append(
            {
                "video_id": video_id,
                "output": str(output),
                "metadata": metadata,
            }
        )
    _emit_json(
        {
            "method": "NOLA preprocessing",
            "processed_videos": len(outputs),
            "outputs": outputs,
        },
        arguments,
    )


def _run_nola(arguments: argparse.Namespace) -> None:
    from pyclad.video import (
        NolaBenchmarkRunner,
        NolaContinualDataset,
        NolaPreparedTestDataset,
        NolaVideoModel,
    )

    continual = NolaContinualDataset(
        arguments.data_root,
        frame_stride=arguments.frame_stride,
    )
    concepts = continual.training_concepts(
        stages=_csv(arguments.stages),
        max_videos_per_stage=_limit(arguments.videos_per_stage),
        max_frames_per_video=_limit(arguments.frames_per_video),
    )
    test = NolaPreparedTestDataset(
        arguments.processed_test_root,
        arguments.ground_truth,
        source_test_root=Path(arguments.data_root) / "Test",
        video_ids=None if arguments.video_ids is None else _csv(arguments.video_ids),
    )

    def model():
        return NolaVideoModel(
            layout=continual.layout,
            neighbors=arguments.neighbors,
            apply_odit=arguments.odit,
            drift=arguments.drift,
        )

    strategy = _nola_strategy(
        arguments.strategy,
        model,
        buffer_size=arguments.buffer_size,
    )
    learn_kwargs = {}
    predict_kwargs = {}
    if arguments.strategy == "mste":
        learn_kwargs = {
            concept.name: {"concept_id": concept.name}
            for concept in concepts
        }
        predict_kwargs = {"concept_id": concepts[-1].name}
    result = NolaBenchmarkRunner().run(
        test,
        strategy,
        train_concepts=concepts,
        learn_kwargs=learn_kwargs,
        predict_kwargs=predict_kwargs,
    )
    _emit_json(
        {
            "method": "NOLA",
            "strategy": result.strategy_name,
            "train_stages": [concept.name for concept in concepts],
            "train_videos": _training_video_records(concepts),
            "train_unique_video_ids": _unique_training_video_ids(concepts),
            "train_rows": sum(len(concept.features) for concept in concepts),
            "test_videos": len(result.frame_scores),
            "frame_metrics": result.frame_metrics.as_dict(),
            "APD": result.average_precision_delay.score,
        },
        arguments,
    )


def _command_strategy(
    name: str,
    model_factory: Callable,
    *,
    epochs: int,
    batch_size: int,
    buffer_size: int,
    device: str,
):
    from pyclad.strategies.baselines.cumulative import CumulativeStrategy
    from pyclad.strategies.baselines.mste import MSTE
    from pyclad.strategies.baselines.naive import NaiveStrategy
    from pyclad.strategies.regularization.der import DerPlusPlus
    from pyclad.strategies.regularization.ewc import EWCStrategy
    from pyclad.strategies.regularization.lwf import LwFStrategy
    from pyclad.strategies.replay.agem import AGEMStrategy
    from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
    from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer
    from pyclad.strategies.replay.replay import ReplayEnhancedStrategy, ReplayOnlyStrategy
    from pyclad.strategies.replay.selection.random import RandomSelection

    if name == "naive":
        return NaiveStrategy(model_factory())
    if name == "cumulative":
        return CumulativeStrategy(model_factory())
    if name == "mste":
        return MSTE(model_factory)
    if name in {"replay-only", "replay-enhanced"}:
        buffer = AdaptiveBalancedReplayBuffer(RandomSelection(), max_size=buffer_size)
        strategy = ReplayOnlyStrategy if name == "replay-only" else ReplayEnhancedStrategy
        return strategy(model_factory(), buffer)
    if name == "ewc":
        return EWCStrategy(
            model_factory(),
            epochs=epochs,
            batch_size=batch_size,
            fisher_batch_size=batch_size,
        )
    if name == "lwf":
        return LwFStrategy(
            model_factory(),
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
    if name == "agem":
        return AGEMStrategy(
            model_factory(),
            ReservoirBuffer(max_capacity=buffer_size),
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
    if name == "der++":
        return DerPlusPlus(
            model=model_factory(),
            buffer=ReservoirBuffer(max_capacity=buffer_size),
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
    raise ValueError(f"unknown COMMAND strategy: {name}")


def _nola_strategy(
    name: str,
    model_factory: Callable,
    *,
    buffer_size: int,
):
    from pyclad.strategies.baselines.cumulative import CumulativeStrategy
    from pyclad.strategies.baselines.mste import MSTE
    from pyclad.strategies.baselines.naive import NaiveStrategy
    from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
    from pyclad.strategies.replay.replay import ReplayEnhancedStrategy, ReplayOnlyStrategy
    from pyclad.strategies.replay.selection.random import RandomSelection

    if name == "naive":
        return NaiveStrategy(model_factory())
    if name == "cumulative":
        return CumulativeStrategy(model_factory())
    if name == "mste":
        return MSTE(model_factory)
    if name in {"replay-only", "replay-enhanced"}:
        buffer = AdaptiveBalancedReplayBuffer(RandomSelection(), max_size=buffer_size)
        strategy = ReplayOnlyStrategy if name == "replay-only" else ReplayEnhancedStrategy
        return strategy(model_factory(), buffer)
    raise ValueError(f"unknown NOLA strategy: {name}")


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


def _training_video_records(concepts: Sequence[Any]) -> int:
    return sum(
        len(
            {
                window.payload.get("record_index", window.video_id)
                for window in concept.windows
            }
        )
        for concept in concepts
    )


def _unique_training_video_ids(concepts: Sequence[Any]) -> int:
    return len(
        {
            window.video_id
            for concept in concepts
            for window in concept.windows
        }
    )


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
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in sorted(vars(arguments).items())
    }


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
