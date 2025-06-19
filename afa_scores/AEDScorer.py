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
        self._stats_numeric = collections.defaultdict(lambda: collections.defaultdict(list))
        self._stats_categorical = collections.defaultdict(lambda: collections.defaultdict(list))

    def learn_one(self, x, y):
        # Add the new instance to the window. The deque will handle the size limit.
        self.window.append((x, y))
        # Recalculate all stats based on the updated window content.
        self._update_stats()

    def _update_stats(self):
        """
        Recalculates all statistics from scratch based on the instances currently in the window.
        """
        # Reset all statistics before recalculating
        self._stats_numeric.clear()
        self._stats_categorical.clear()
        self.feature_types.clear()

        # Iterate over all instances in the current window to build up stats
        for inst_x, inst_y in self.window:
            for feature, value in inst_x.items():
                if value is None or pd.isna(value):
                    continue

                if isinstance(value, (int, float)):
                    if feature not in self.feature_types:
                        self.feature_types[feature] = 'numeric'
                    if self.feature_types[feature] == 'numeric':
                        self._stats_numeric[inst_y][feature].append(value)
                else:
                    if feature not in self.feature_types:
                        self.feature_types[feature] = 'categorical'
                    if self.feature_types[feature] == 'categorical':
                        self._stats_categorical[inst_y][feature].append(value)

    def _calculate_numeric_aed(self, feature: str) -> float:
        class_labels = list(self._stats_numeric.keys())
        total_dist_sq = 0.0
        pairs = 0
        for i in range(len(class_labels)):
            for j in range(i + 1, len(class_labels)):
                c1, c2 = class_labels[i], class_labels[j]

                # Get lists of values for the feature for each class
                values1 = self._stats_numeric[c1].get(feature)
                values2 = self._stats_numeric[c2].get(feature)

                if values1 and values2:
                    mean1 = np.mean(values1)
                    mean2 = np.mean(values2)
                    total_dist_sq += (mean1 - mean2) ** 2
                    pairs += 1
        return math.sqrt(total_dist_sq) if pairs > 0 else 0.0

    def _calculate_categorical_aed(self, feature: str) -> float:
        class_labels = list(self._stats_categorical.keys())
        total_aed_nom = 0.0

        for i in range(len(class_labels)):
            for j in range(i + 1, len(class_labels)):
                c1, c2 = class_labels[i], class_labels[j]

                values1 = self._stats_categorical[c1].get(feature)
                values2 = self._stats_categorical[c2].get(feature)

                if values1 and values2:
                    counts1 = collections.Counter(values1)
                    counts2 = collections.Counter(values2)
                    all_values = set(counts1.keys()) | set(counts2.keys())
                    n_values = len(all_values)

                    if n_values == 0: continue

                    total1 = len(values1)
                    total2 = len(values2)

                    if total1 == 0 or total2 == 0: continue

                    inner_sum = sum(
                        abs(counts1.get(val, 0) / total1 - counts2.get(val, 0) / total2)
                        for val in all_values
                    )
                    total_aed_nom += (1 / n_values) * inner_sum

        return total_aed_nom

    def get_merits(self, feature_names: list, feature_costs: dict) -> dict:
        aed_scores = {}
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

