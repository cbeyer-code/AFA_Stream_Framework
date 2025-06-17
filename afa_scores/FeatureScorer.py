import abc
from river import base

class FeatureScorer(base.Base, abc.ABC):
    """Abstract base class for feature scoring methods."""

    @abc.abstractmethod
    def get_merits(self, feature_names: list, feature_costs: dict) -> dict:
        """Calculates the merit for each feature."""
        pass

    @abc.abstractmethod
    def get_global_merits(self) -> dict:
        """Returns all calculated merits for global ranking."""
        pass

    @abc.abstractmethod
    def learn_one(self, x: dict, y: base.typing.ClfTarget):
        """Updates the scorer's internal state."""
        pass