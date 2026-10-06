"""Archive annotation and task-allocation regressions."""

import numpy as np
import pytest

from pyclad.video.data.command_ucf_crime import CommandUcfCrimeDataset
from pyclad.video.metrics.frame_score_utils import window_scores_to_frame_scores


@pytest.fixture
def archive(tmp_path):
    anomalies = [f"{category}/train{index}.mp4" for category in ("Abuse", "Burglary", "Robbery") for index in range(2)]
    normals = [f"Normal_Videos_event/train{index}.mp4" for index in range(6)]
    test = ["Normal_Videos_event/test.mp4", "Abuse/test.mp4"]
    for stream in ("all_rgbs", "all_flows"):
        for name in anomalies + normals + test:
            path = tmp_path / stream / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.zeros((32, 1024), dtype=np.float32))
    (tmp_path / "train_normal.txt").write_text("\n".join(normals) + "\n")
    (tmp_path / "train_anomaly.txt").write_text("\n".join(anomalies) + "\n")
    (tmp_path / "test_normalv2.txt").write_text(f"{test[0]} 67 -1\n")
    (tmp_path / "test_anomalyv2.txt").write_text(f"{test[1]}|67|[1, 1, 33, 48, 65, 68]\n")
    return tmp_path


def test_annotations_and_uneven_windows_align_on_frames(archive):
    concept = CommandUcfCrimeDataset(archive).read_dataset().test_concepts()[0]
    bag = concept.data[1]
    expected_labels = np.zeros(67, dtype=np.int64)
    expected_labels[0] = 1
    expected_labels[32:48] = 1
    expected_labels[64:] = 1
    np.testing.assert_array_equal(concept.frame_labels[bag.bag_id], expected_labels)

    # The release contains intervals ending one frame beyond the video; retain clipping.
    scores = np.arange(32, dtype=np.float64)
    curves = window_scores_to_frame_scores(bag.windows, scores, {bag.bag_id: 67})
    widths = [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2]
    np.testing.assert_array_equal(curves[bag.bag_id], np.repeat(scores, widths))
    assert bag.windows[0].start_frame == 0
    assert bag.windows[-1].end_frame == 66


def test_limit_preserves_normal_task_allocation_and_test_split(archive):
    reader = CommandUcfCrimeDataset(archive)
    full = reader.read_dataset()
    limited = reader.read_dataset(max_videos_per_class=1)
    normal_ids = []
    for complete, small in zip(full.train_concepts(), limited.train_concepts()):
        assert complete.name == small.name
        assert len(complete.data) == 4
        assert len(small.data) == 2
        full_normals = [bag.bag_id for bag in complete.data if bag.weak_label == 0]
        small_normals = [bag.bag_id for bag in small.data if bag.weak_label == 0]
        assert small_normals == full_normals[:1]
        normal_ids.extend(full_normals)
    assert len(set(normal_ids)) == 6
    assert [bag.bag_id for bag in full.test_concepts()[0].data] == [
        bag.bag_id for bag in limited.test_concepts()[0].data
    ]


@pytest.mark.parametrize("boundaries", ["[-1, -1]", "[0, 4]", "[8, 7]"])
def test_invalid_annotations_cannot_silently_change_frame_labels(archive, boundaries):
    (archive / "test_anomalyv2.txt").write_text(f"Abuse/test.mp4|67|{boundaries}\n")
    with pytest.raises(ValueError, match="positive, ordered frame coordinates"):
        CommandUcfCrimeDataset(archive)


@pytest.mark.parametrize("custom_cache", [False, True])
def test_hub_download_loads_paired_features_and_original_tasks(archive, monkeypatch, custom_cache):
    """The downloaded snapshot must feed the same RGB-first bag reader as local data."""
    from fnmatch import fnmatch

    from pyclad.video.data import command_ucf_crime

    for stream, value in (("all_rgbs", 1.0), ("all_flows", 2.0)):
        np.save(archive / stream / "Abuse/train0.mp4.npy", np.full((32, 1024), value, dtype=np.float32))
    options = dict(cache_dir=archive / "cache", revision="test-revision", local_files_only=True) if custom_cache else {}

    def download(*, repo_id, repo_type, cache_dir, revision, local_files_only, allow_patterns):
        assert repo_id == "nmcnamee24/ucf-crime-rgb-flow-features"
        assert repo_type == "dataset"
        assert cache_dir == options.get("cache_dir")
        assert revision == options.get("revision")
        assert local_files_only == options.get("local_files_only", False)
        for path in archive.rglob("*"):
            if path.is_file():
                assert any(fnmatch(path.relative_to(archive).as_posix(), pattern) for pattern in allow_patterns)
        assert not any(fnmatch("unrelated.zip", pattern) for pattern in allow_patterns)
        return str(archive)

    monkeypatch.setattr(command_ucf_crime, "snapshot_download", download, raising=False)
    dataset = CommandUcfCrimeDataset(**options)
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [4, 4, 4]
    assert len(dataset.test_concepts()[0].data) == 2
    bag = dataset.train_concepts()[0].data[0]
    assert bag.features.shape == (32, 2048)
    np.testing.assert_array_equal(bag.features[:, :1024], 1.0)
    np.testing.assert_array_equal(bag.features[:, 1024:], 2.0)


def test_constructor_returns_ready_concepts_like_core_loaders(archive):
    """A freshly constructed loader exposes concepts and core output metadata."""
    dataset = CommandUcfCrimeDataset(archive)
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [4, 4, 4]
    assert len(dataset.test_concepts()[0].data) == 2
    assert dataset.info() == {"dataset": {"name": "COMMAND-UCF-Crime", "tran_concepts_no": 3, "test_concepts_no": 1}}
    assert dataset.read_dataset() is dataset


def test_constructor_can_limit_training_without_changing_test_split(archive):
    """Small runs use the same dataset contract and retain all test videos."""
    dataset = CommandUcfCrimeDataset(archive, max_videos_per_class=1)
    assert [len(c.data) for c in dataset.train_concepts()] == [2, 2, 2]
    assert len(dataset.test_concepts()[0].data) == 2


def test_explicit_local_root_never_downloads(archive, monkeypatch):
    """Local archives, including incomplete ones, must never fall back to the Hub."""
    from pyclad.video.data import command_ucf_crime

    def unexpected_download(**kwargs):
        pytest.fail("an explicit local root must not contact Hugging Face")

    monkeypatch.setattr(command_ucf_crime, "snapshot_download", unexpected_download, raising=False)
    dataset = CommandUcfCrimeDataset(archive, cache_dir=archive / "unused").read_dataset()
    assert [len(c.data) for c in dataset.train_concepts()] == [4, 4, 4]
    with pytest.raises(FileNotFoundError, match="layout is incomplete"):
        CommandUcfCrimeDataset(archive / "missing")


def test_hub_download_failure_preserves_actionable_error(monkeypatch):
    """Do not turn a failed download into a misleading local-layout error."""
    from pyclad.video.data import command_ucf_crime

    def failed_download(**kwargs):
        raise ConnectionError("Cannot reach Hugging Face; retry or use a cached snapshot")

    monkeypatch.setattr(command_ucf_crime, "snapshot_download", failed_download, raising=False)
    with pytest.raises(ConnectionError, match="Cannot reach Hugging Face"):
        CommandUcfCrimeDataset()


@pytest.mark.longrun
def test_downloaded_release_and_offline_cache_have_all_three_tasks(tmp_path):
    """Exercise the published paired release through the public dataset loader."""
    options = dict(cache_dir=tmp_path, revision="5b33de08561dc49378678ab5c696198b593aaa6c")
    reader = CommandUcfCrimeDataset(**options)
    dataset = reader.read_dataset()
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [362, 576, 682]
    assert len(dataset.test_concepts()[0].data) == 290
    for concept in dataset.train_concepts() + dataset.test_concepts():
        assert all(bag.features.shape == (32, 2048) and np.isfinite(bag.features).all() for bag in concept.data)
    cached = CommandUcfCrimeDataset(**options, local_files_only=True)
    assert cached.root == reader.root
    limited = cached.read_dataset(max_videos_per_class=1)
    assert [len(c.data) for c in limited.train_concepts()] == [8, 8, 10]
    np.testing.assert_array_equal(
        limited.train_concepts()[0].data[0].features, dataset.train_concepts()[0].data[0].features
    )
