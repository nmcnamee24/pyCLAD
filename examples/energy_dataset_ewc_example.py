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
from pyclad.strategies.regularization.ewc import EWCStrategy

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example showcase how to run a concept aware scenario using the Energy dataset adopted to continual anomaly
    detection using the method proposed here <https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios>
    """
    dataset = EnergyPlantsDataset(dataset_type="random_anomalies")
    input_features = 14

    print(input_features)

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

    model = Autoencoder(encoder, decoder)

    # Reconstruction losses often need a much larger lambda than the generic EWC default.
    strategy = EWCStrategy(
        model,
        module=model.module,
        loss_fn=EWCStrategy.default_loss_fn,
        data_transform=EWCStrategy.identity_transform,
        batch_size=32,
        lr=model.module.lr,
        epochs=30,
        device=model.device,
        shuffle=True,
        ewc_lambda=10000.0,
        fisher_estimation_mode="eval",
        constraint_retention="all",
    )
    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]
    scenario = ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output-energy-EWC.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
