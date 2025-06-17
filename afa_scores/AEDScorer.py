from .FeatureScorer import FeatureScorer
import collections
import numpy as np
import math

class AEDScorer(FeatureScorer):
    """
    Scores features based on Average Euclidean Distance (AED) for class separation.
    Merit is defined as AED(feature) / cost(feature).
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.window = collections.deque(maxlen=window_size)
        self._merits = {}
        self._stats = {}  # {class: {feature: (mean, variance)}}

    def learn_one(self, x, y):
        self.window.append((x, y))
        self._update_stats()

    def _update_stats(self):
        # In a real high-performance scenario, these stats would be updated incrementally.
        # For simplicity here, we recalculate over the window.
        stats_by_class = collections.defaultdict(lambda: collections.defaultdict(list))
        for x, y in self.window:
            for feature, value in x.items():
                if isinstance(value, (int, float)):  # Only for numeric features
                    stats_by_class[y][feature].append(value)

        self._stats = collections.defaultdict(dict)
        for y, features in stats_by_class.items():
            for feature, values in features.items():
                if len(values) > 1:
                    self._stats[y][feature] = (np.mean(values), np.var(values))

    def get_merits(self, feature_names: list, feature_costs: dict) -> dict:
        if not self._stats:
            return {name: 0.0 for name in feature_names}

        aed_scores = {}
        class_labels = list(self._stats.keys())

        for feature in feature_names:
            total_dist = 0.0
            pairs = 0
            # Calculate pairwise AED for the feature across all classes
            for i in range(len(class_labels)):
                for j in range(i + 1, len(class_labels)):
                    c1, c2 = class_labels[i], class_labels[j]
                    if feature in self._stats[c1] and feature in self._stats[c2]:
                        mean1, _ = self._stats[c1][feature]
                        mean2, _ = self._stats[c2][feature]
                        total_dist += (mean1 - mean2) ** 2
                        pairs += 1

            aed = math.sqrt(total_dist / pairs) if pairs > 0 else 0
            cost = feature_costs.get(feature, 1)
            aed_scores[feature] = aed / cost if cost > 0 else aed

        self._merits = aed_scores
        return self._merits

    def get_global_merits(self) -> dict:
        return self._merits