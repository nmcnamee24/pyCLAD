import numpy as np
from numpy.testing import assert_array_equal

from pyclad.strategies.replay.buffers.balanced import BalancedReplayBuffer


def test_updating_balanced_buffer_stores_examples_and_concept_indices():
    buffer = BalancedReplayBuffer(max_size=10, seed=0)
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)

    buffer.update(data)

    arrays = buffer.arrays()
    assert_array_equal(arrays["examples"], data)
    assert_array_equal(arrays["concept_indices"], np.array([0, 0], dtype=np.int64))


def test_sampling_balanced_buffer_returns_all_registered_fields():
    buffer = BalancedReplayBuffer(max_size=10, seed=0)
    examples = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    concept_indices = np.array([0, 0, 1], dtype=np.int64)
    losses = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    buffer.add(
        examples=examples,
        concept_indices=concept_indices,
        reconstruction_loss=losses,
    )

    sample = buffer.sample(2)

    assert set(sample) == {"examples", "concept_indices", "reconstruction_loss"}
    assert sample["examples"].shape == (2, 2)
    assert sample["concept_indices"].shape == (2,)
    assert sample["reconstruction_loss"].shape == (2,)


def test_rebalancing_balanced_buffer_drops_from_largest_concept():
    buffer = BalancedReplayBuffer(max_size=4, seed=0)
    first_concept = np.array([[0], [1], [2], [3]], dtype=np.float32)
    second_concept = np.array([[10], [11], [12], [13]], dtype=np.float32)

    buffer.add(first_concept, np.zeros(len(first_concept), dtype=np.int64))
    buffer.add(second_concept, np.ones(len(second_concept), dtype=np.int64))

    concept_indices = buffer.arrays()["concept_indices"]
    assert len(buffer.data()) == 4
    assert np.count_nonzero(concept_indices == 0) == 2
    assert np.count_nonzero(concept_indices == 1) == 2
