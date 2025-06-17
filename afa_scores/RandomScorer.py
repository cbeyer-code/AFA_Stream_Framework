import random
from river import base
from .FeatureScorer import FeatureScorer

class RandomScorer(FeatureScorer):
    """Assigns a random score to each feature."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._merits = {}

    def get_merits(self, feature_names: list, feature_costs: dict) -> dict:
        self._merits = {name: self.rng.random() for name in feature_names}
        return self._merits

    def get_global_merits(self) -> dict:
        return self._merits

    def learn_one(self, x: dict, y: base.typing.ClfTarget):
        # No learning needed for random scorer
        pass

