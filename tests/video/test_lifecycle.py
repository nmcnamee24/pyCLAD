"""Full archive-to-scenario coverage for the single COMMAND workflow."""

import json

import numpy as np
import pytest
import torch

from pyclad.video.data.command_ucf_crime import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    CommandUcfCrimeDataset,
)
from pyclad.video.models.command.command import CommandModel
from pyclad.video.models.command.config import (
    CommandArchitectureConfig,
    CommandTrainerConfig,
)


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


def test_three_tasks_use_core_model_scenario_metrics_and_writer(archive):
    from pyclad.models.model import Model
    from pyclad.output.json_writer import JsonOutputWriter
    from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
    from pyclad.video.callbacks.video_frame_metric_callback import (
        VideoFrameMetricCallback,
    )
    from pyclad.video.metrics.frame_roc_auc import FrameRocAuc
    from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy

    torch.manual_seed(11)
    architecture = CommandArchitectureConfig(
        hidden_dim=8, state_dim=3, mamba_blocks=1, memory_size=5, projection_dim=6, dropout=0.0
    )
    config = CommandTrainerConfig(
        architecture=architecture, epochs=1, batch_size=4, replay_batch_size=2, buffer_size=6, device="cpu"
    )
    model = CommandModel(config=config)
    assert isinstance(model, Model)
    dataset = CommandUcfCrimeDataset(archive).read_dataset()
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [8, 8, 10]
    strategy = ContTrainPlusPlusStrategy(model)
    callback = VideoFrameMetricCallback(FrameRocAuc())
    ConceptIncrementalScenario(dataset, strategy, [callback]).run()
    assert len(model.replay) == 6
    info = callback.info()["concept_metric_callback_Frame-ROC-AUC"]
    assert list(info["metric_matrix"]) == ["T1", "T2", "T3"]
    assert all(0 <= row["test"] <= 1 for row in info["metric_matrix"].values())
    output = archive / "result.json"
    JsonOutputWriter(output).write([dataset, model, strategy, callback])
    assert json.loads(output.read_text())["model"]["name"] == "COMMAND"
    model.save_checkpoint(archive / "model.pt")
    restored = CommandModel(config=config)
    restored.load_checkpoint(archive / "model.pt")
    test_data = dataset.test_concepts()[0].data
    for training in (True, False):
        model.model.train(training)
        prediction = model.predict(test_data)
        assert model.model.training is training
        np.testing.assert_array_equal(prediction.window_scores, restored.predict(test_data).window_scores)


def test_duplicate_training_rows_keep_distinct_bags(archive):
    path = archive / "train_normal.txt"
    lines = path.read_text().splitlines()
    lines[1] = lines[0]
    path.write_text("\n".join(lines) + "\n")
    dataset = CommandUcfCrimeDataset(archive).read_dataset()
    bags = [bag for c in dataset.train_concepts() for bag in c.data]
    assert len(bags) == len({b.bag_id for b in bags}) == 26
    assert all(b.features.shape == (32, 2048) for b in bags)
    labels = dataset.test_concepts()[0].frame_labels
    assert labels["Abuse/test.mp4"].sum() == 16


@pytest.mark.parametrize("fault", ["overlap", "shape", "nonfinite", "balance"])
def test_reader_rejects_invalid_archive(archive, fault):
    if fault == "overlap":
        (archive / "test_normalv2.txt").write_text("Normal_Videos_event/train0.mp4 64 -1\n")
    elif fault == "balance":
        (archive / "train_normal.txt").write_text("")
    else:
        values = np.zeros((2, 1024)) if fault == "shape" else np.full((32, 1024), np.nan)
        np.save(archive / "all_rgbs/Abuse/train.mp4.npy", values)
    with pytest.raises(ValueError):
        CommandUcfCrimeDataset(archive).read_dataset()


def test_training_rejects_mixed_tasks_and_row_matrices(archive):
    dataset = CommandUcfCrimeDataset(archive).read_dataset()
    model = CommandModel(
        config=CommandTrainerConfig(architecture=CommandArchitectureConfig(hidden_dim=8, memory_size=5), epochs=1)
    )
    with pytest.raises(ValueError, match="exactly one task"):
        model.fit(np.concatenate([c.data for c in dataset.train_concepts()]))
    with pytest.raises(ValueError, match="VideoBag"):
        model.fit(np.zeros((3, 2048)))
