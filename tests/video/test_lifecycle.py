"""End-to-end coverage of video integration with core scenario contracts."""

import json

import numpy as np
import pytest
import torch

from pyclad.video.cli import main
from pyclad.video.datasets.command_ucf_crime import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    CommandUcfCrimeDataset,
)
from pyclad.video.models.command.model import CommandVideoModel
from pyclad.video.models.command.paper_architecture import (
    PaperCommandArchitectureConfig,
)
from pyclad.video.models.command.paper_training import PaperCommandTrainerConfig


@pytest.fixture
def archive(tmp_path):
    """Make disjoint 32-window RGB/flow train and test splits."""
    anomalies = [f"{category}/train.mp4" for category in COMMAND_UCF_CRIME_CONCEPT_ORDER]
    normals = [f"Normal_Videos_event/train{index}.mp4" for index in range(13)]
    test = ["Normal_Videos_event/test.mp4", "Abuse/test.mp4"]
    rng = np.random.default_rng(19)
    for stream in ("all_rgbs", "all_flows"):
        for name in anomalies + normals + test:
            path = tmp_path / stream / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, rng.normal(size=(32, 1024)).astype(np.float32))
    (tmp_path / "train_normal.txt").write_text("\n".join(normals) + "\n")
    (tmp_path / "train_anomaly.txt").write_text("\n".join(anomalies) + "\n")
    (tmp_path / "test_normalv2.txt").write_text(f"{test[0]} 64 -1\n")
    (tmp_path / "test_anomalyv2.txt").write_text(f"{test[1]}|64|[33, 48]\n")
    return tmp_path


def test_cli_runs_three_tasks_through_core_scenario(archive, monkeypatch, capsys):
    """Train, evaluate and checkpoint all tasks using small actual networks."""
    from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario

    torch.manual_seed(11)
    architecture = PaperCommandArchitectureConfig(
        hidden_dim=8,
        state_dim=3,
        mamba_blocks=1,
        memory_size=5,
        projection_dim=6,
        dropout=0.0,
    )
    config = PaperCommandTrainerConfig(
        architecture=architecture,
        epochs=1,
        batch_size=4,
        replay_batch_size=2,
        buffer_size=6,
        device="cpu",
    )
    monkeypatch.setattr("pyclad.video.cli._paper_trainer_config", lambda _: config)
    calls = []
    run = ConceptIncrementalScenario.run

    def observed_run(self):
        calls.append(self)
        return run(self)

    monkeypatch.setattr(ConceptIncrementalScenario, "run", observed_run)
    main(
        [
            "ucf-command",
            "--data-root",
            str(archive),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--checkpoint-root",
            str(archive / "checkpoints"),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert len(calls) == 1
    assert result["protocol"]["continual_tasks"] == ["T1", "T2", "T3"]
    assert result["train_manifest_records"] == 26
    assert result["test_videos"] == 2
    assert len(result["task_results"]) == 3
    for task in result["task_results"]:
        assert task["training"]["buffer_bags"] == 6
        assert 0 <= task["evaluation"]["ddm"]["AUC"] <= 1
        state = torch.load(task["checkpoint"], map_location="cpu", weights_only=False)
        assert state
    assert result["validation"]["finite"]


@pytest.mark.parametrize("tasks", [(), ("T2", "T1"), ("T1", "T1"), ("unknown",)])
def test_reader_rejects_ambiguous_task_selection(archive, tasks):
    with pytest.raises(ValueError, match="tasks must"):
        CommandUcfCrimeDataset(archive).read_dataset(tasks=tasks)


@pytest.mark.parametrize("training", [True, False])
def test_prediction_restores_mode_and_uses_current_device(training, monkeypatch):
    model = CommandVideoModel(4, hidden_dim=8, embedding_dim=6, memory_size=4)
    model.get_module().train(training)
    calls = []
    device = model.device

    def observed_device():
        calls.append(True)
        return device()

    monkeypatch.setattr(model, "device", observed_device)
    result = model.predict(np.ones((3, 4), dtype=np.float32))
    assert calls
    assert model.get_module().training is training
    assert result.anomaly_scores.shape == (3,)
    model.to(torch.device("meta"))
    assert model.additional_info()["device"] == "meta"
