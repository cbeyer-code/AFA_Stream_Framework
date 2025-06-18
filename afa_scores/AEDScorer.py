import pandas as pd
from river import stats
from .FeatureScorer import FeatureScorer
import collections
import numpy as np
import math


class AEDScorer(FeatureScorer):
    """
    Scores features based on Average Euclidean Distance (AED) for class separation.
    - For numeric features, it's the distance between class means.
    - For categorical features, it's the distance between class probability distributions
      as defined in "Active Feature Acquisition in Data Streams" (Beyer et al., ECML PKDD 2020).
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.window = collections.deque(maxlen=window_size)
        self._merits = {}
        self.feature_types = {}
        self._stats_numeric = collections.defaultdict(lambda: collections.defaultdict(stats.Mean))
        self._stats_categorical = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

    def learn_one(self, x, y):
        self.window.append((x, y))
        self._update_stats()

    def _update_stats(self):
        # This is inefficient but simple; recalculates stats over the full window.
        # A more performant version would incrementally update/remove stats.
        self._stats_numeric.clear()
        self._stats_categorical.clear()
        self.feature_types.clear()

        for inst_x, inst_y in self.window:
            for feature, value in inst_x.items():
                # Skip missing values in our statistical calculations
                if value is None or pd.isna(value):
                    continue

                if isinstance(value, (int, float)):
                    if feature not in self.feature_types: self.feature_types[feature] = 'numeric'
                    # Only update if the type is consistent
                    if self.feature_types[feature] == 'numeric':
                        self._stats_numeric[inst_y][feature].update(value)
                else:  # Treat everything else as categorical
                    if feature not in self.feature_types: self.feature_types[feature] = 'categorical'
                    if self.feature_types[feature] == 'categorical':
                        self._stats_categorical[inst_y][feature].update([value])

    def _calculate_numeric_aed(self, feature: str) -> float:
        class_labels = list(self._stats_numeric.keys())
        total_dist_sq = 0.0
        pairs = 0
        for i in range(len(class_labels)):
            for j in range(i + 1, len(class_labels)):
                c1, c2 = class_labels[i], class_labels[j]
                if feature in self._stats_numeric[c1] and feature in self._stats_numeric[c2]:
                    mean1 = self._stats_numeric[c1][feature].get()
                    mean2 = self._stats_numeric[c2][feature].get()
                    if mean1 is not None and mean2 is not None:
                        total_dist_sq += (mean1 - mean2) ** 2
                        pairs += 1
        return math.sqrt(total_dist_sq) if pairs > 0 else 0.0

    def _calculate_categorical_aed(self, feature: str) -> float:
        class_labels = list(self._stats_categorical.keys())
        total_aed_nom = 0.0

        # Outer sum over all pairs of classes (c, k)
        for i in range(len(class_labels)):
            for j in range(i + 1, len(class_labels)):
                c1, c2 = class_labels[i], class_labels[j]

                counts1 = self._stats_categorical[c1].get(feature)
                counts2 = self._stats_categorical[c2].get(feature)

                if counts1 and counts2:
                    # V: union of all possible values for this feature across both classes
                    all_values = set(counts1.keys()) | set(counts2.keys())

                    # |V|
                    n_values = len(all_values)
                    if n_values == 0:
                        continue

                    total1 = sum(counts1.values())
                    total2 = sum(counts2.values())

                    if total1 == 0 or total2 == 0:
                        continue

                    # Inner sum: sum_{v in V} |P(v|c) - P(v|k)|
                    # The paper's formula is sqrt((P(v|c) - P(v|k))^2) which is abs(P(v|c) - P(v|k))
                    inner_sum = sum(
                        abs(counts1.get(val, 0) / total1 - counts2.get(val, 0) / total2)
                        for val in all_values
                    )

                    # Add the term for this class pair: (1/|V|) * inner_sum
                    total_aed_nom += (1 / n_values) * inner_sum

        return total_aed_nom

    def get_merits(self, feature_names: list, feature_costs: dict) -> dict:
        aed_scores = {}
        # Get merits for all features, not just missing ones, so we have a complete picture for plotting
        all_feature_names = list(self.feature_types.keys())
        for feature in all_feature_names:
            feature_type = self.feature_types.get(feature)
            aed = 0.0
            if feature_type == 'numeric':
                aed = self._calculate_numeric_aed(feature)
            elif feature_type == 'categorical':
                aed = self._calculate_categorical_aed(feature)

            cost = feature_costs.get(feature, 1)
            aed_scores[feature] = aed / cost if cost > 0 else aed

        self._merits = aed_scores
        return self._merits

    def get_global_merits(self) -> dict:
        return self._merits
