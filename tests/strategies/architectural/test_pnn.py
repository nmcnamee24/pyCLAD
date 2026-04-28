import importlib

import numpy as np
import torch
from torch import nn

from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.strategies import PNNStrategy
from pyclad.strategies.architectural import PNNStrategy as ArchitecturalPNNStrategy


def make_tabular_base_model():
    encoder = nn.Sequential(nn.Linear(10, 4))
    decoder = nn.Sequential(nn.Linear(4, 10))
    return Autoencoder(encoder=encoder, decoder=decoder, epochs=1)


def make_image_base_model():
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 8),
    )
    decoder = nn.Sequential(
        nn.Linear(8, 64),
        nn.Unflatten(1, (1, 8, 8)),
    )
    return Autoencoder(encoder=encoder, decoder=decoder, epochs=1)


def test_pnn_strategy_imports_from_strategies_package_and_not_models_package():
    autoencoder_package = importlib.import_module("pyclad.models.autoencoder")

    assert PNNStrategy is ArchitecturalPNNStrategy
    assert not hasattr(autoencoder_package, "PNNStrategy")
    assert not hasattr(autoencoder_package, "PNN")


def test_pnn_basic_tabular_fit_predict():
    data = np.random.randn(64, 10).astype(np.float32)
    strategy = PNNStrategy(base_model_factory=make_tabular_base_model, random_state=0)

    strategy.fit(data)
    predictions, scores = strategy.predict(data)

    assert predictions.shape == (64,)
    assert scores.shape == (64,)


def test_pnn_end_task_creates_new_column_and_freezes_old_one():
    strategy = PNNStrategy(base_model_factory=make_tabular_base_model, random_state=0)

    assert strategy.num_columns == 1

    strategy.end_task()

    assert strategy.num_columns == 2
    assert all(not parameter.requires_grad for parameter in strategy.module.columns[0].parameters())
    assert all(parameter.requires_grad for parameter in strategy.module.columns[1].parameters())


def test_pnn_task_free_prediction_uses_minimum_reconstruction_error():
    data = np.random.randn(64, 10).astype(np.float32)
    shifted = data + 1.0
    strategy = PNNStrategy(base_model_factory=make_tabular_base_model, task_free=True, random_state=0)

    strategy.fit(data)
    strategy.end_task()
    strategy.fit(shifted)

    _, task0_scores = strategy.predict(shifted, task_label=0)
    _, task1_scores = strategy.predict(shifted, task_label=1)
    _, task_free_scores = strategy.predict(shifted)

    assert task_free_scores.shape == (64,)
    assert np.allclose(task_free_scores, np.minimum(task0_scores, task1_scores))


def test_pnn_task_free_ignores_concept_routing():
    data = np.random.randn(64, 10).astype(np.float32)
    shifted = data + 1.0
    strategy = PNNStrategy(base_model_factory=make_tabular_base_model, task_free=True, random_state=0)

    strategy.learn(data, concept_id="concept_0")
    strategy.learn(shifted, concept_id="concept_1")

    _, task_free_scores = strategy.predict(shifted)
    _, concept_routed_scores = strategy.predict(shifted, concept_id="concept_1")

    assert np.allclose(task_free_scores, concept_routed_scores)


def test_pnn_image_like_scores_are_flat_per_sample():
    data = np.random.randn(16, 1, 8, 8).astype(np.float32)
    strategy = PNNStrategy(base_model_factory=make_image_base_model, random_state=0)

    strategy.fit(data)
    _, scores = strategy.predict(data)

    assert scores.shape == (16,)


def test_pnn_training_new_column_does_not_modify_old_column_parameters():
    data = np.random.randn(32, 10).astype(np.float32)
    shifted = data + 1.0
    strategy = PNNStrategy(base_model_factory=make_tabular_base_model, random_state=0)

    strategy.fit(data)
    strategy.end_task()

    old_parameters = [parameter.detach().clone() for parameter in strategy.module.columns[0].parameters()]
    strategy.fit(shifted)

    for saved_parameter, current_parameter in zip(old_parameters, strategy.module.columns[0].parameters()):
        assert torch.equal(saved_parameter, current_parameter.detach())
