"""Expose COMMAND's persistent ContTrain++ state to the core scenario."""

from pyclad.strategies.strategy import ConceptIncrementalStrategy
from pyclad.video.models.command.command import CommandModel


class ContTrainPlusPlusStrategy(ConceptIncrementalStrategy):
    """Delegate complete-bag learning and replay to one COMMAND model."""

    def __init__(self, model: CommandModel):
        self._model = model

    def learn(self, data):
        """Train on the next task while retaining the model's replay memory."""
        self._model.fit(data)

    def predict(self, data):
        """Return temporal scores without using evaluation task identity."""
        return self._model.predict(data)

    def name(self):
        return "ContTrain++"
