import logging
import pathlib

import torch.nn as nn

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies import PNNStrategy

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])


def make_model(input_features: int) -> Autoencoder:
    encoder = nn.Sequential(
        nn.Linear(input_features, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
    )

    decoder = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, input_features),
        nn.Sigmoid(),
    )

    return Autoencoder(encoder, decoder, epochs=30)


if __name__ == "__main__":
    """
    This example showcases how to run a concept aware scenario using the Energy dataset adopted to continual anomaly
    detection using the method proposed here <https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios>.
    PNN expands the autoencoder architecture with a new frozen-aware column for each concept and uses lateral latent
    adapters to transfer knowledge from previous concepts. This example enables task-free inference, so predictions
    use the minimum reconstruction error across all learned columns instead of routing by concept id at test time.
    """
    dataset = EnergyPlantsDataset(dataset_type="random_anomalies")
    input_features = 14

    print(input_features)

    model = make_model(input_features)
    strategy = PNNStrategy(base_model_factory=lambda: make_model(input_features), task_free=True, random_state=42)

    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]
    scenario = ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output-energy-PNN.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
