import abc
from typing import List

ConceptLevelMatrix = List[List[float]] # ConceptLevelMatrix[learned_concept][evaluated_concept]


class ConceptLevelMetric(abc.ABC):
    """Base class for metrics that transform the concept-level metric matrix to summarized single value metric."""
    @abc.abstractmethod
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...


class PerLearnedConceptMetric(abc.ABC):
    """Base class for metrics that transform the concept-level metric matrix a separate value for each learned concept."""
    @abc.abstractmethod
    def compute(self, metric_matrix: ConceptLevelMatrix) -> List[float]: ...

    @abc.abstractmethod
    def name(self) -> str: ...