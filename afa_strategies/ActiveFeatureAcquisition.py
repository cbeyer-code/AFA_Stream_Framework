from typing import List
import numpy as np
import pandas as pd
from river import base

from afa_scores import FeatureScorer
from budget_managers import BudgetManager


class ActiveFeatureAcquisition(base.Transformer):
    """
    A River transformer that performs Active Feature Acquisition.
    This is the core component that orchestrates the AFA process for each instance.
    """

    def __init__(self, scorer: FeatureScorer, budget_manager: BudgetManager,
                 feature_costs: dict, acquisition_strategy: str = 'k-best', k: int = 1):
        self.scorer = scorer
        self.budget_manager = budget_manager
        self.feature_costs = feature_costs
        self.acquisition_strategy = acquisition_strategy
        self.k = k
        self.features_acquired_this_step = 0

    def _calculate_quality(self, features: List[str], merits: dict) -> float:
         """Calculates quality as the mean of merits for a given list of features."""
         feature_merits = [merits.get(f, 0) for f in features if f in merits]
         return np.mean(feature_merits) if feature_merits else 0.0

    def learn_one(self, x, y=None):
        # The AFA transformer is "stateful" via its components (scorer, budget_manager)
        # We update the state of these components using the *true* data.
        _, x_true = x
        self.scorer.learn_one(x_true, y)
        self.budget_manager.learn_one()
        return self

    """ 
    def transform_one_old(self, x):
        # The input x is expected to be a tuple: (x_missing, x_complete)
        x_miss, x_true = x
        self.features_acquired_this_step = 0

        # Identify missing features
        missing_features = [f for f, v in x_miss.items() if v is None or pd.isna(v)]
        if not missing_features:
            return x_miss

        # Get feature merits from the scorer
        merits = self.scorer.get_merits(missing_features, self.feature_costs)
        print(merits)
        # Select acquisition candidates based on strategy
        acquisition_candidates = self._select_candidates(x_miss, merits, missing_features)

        # Perform acquisition
        x_acquired = x_miss.copy()
        # This loop attempts to acquire features one by one from the candidates list
        for feature_to_acquire in acquisition_candidates:
            cost = self.feature_costs.get(feature_to_acquire, 1)

            # Calculate quality gain
            quality_before = self._calculate_quality(x_acquired, merits)

            # Temporarily acquire to calculate quality after
            temp_x = x_acquired.copy()
            temp_x[feature_to_acquire] = x_true[feature_to_acquire]
            quality_after = self._calculate_quality(temp_x, merits)
            quality_gain = quality_after - quality_before

            # Ask budget manager for permission
            if self.budget_manager.can_acquire(quality_gain, cost):
                # If approved, permanently acquire
                x_acquired[feature_to_acquire] = x_true[feature_to_acquire]
                self.features_acquired_this_step += 1

        return x_acquired
    """

    def transform_one(self, x):
        # The input x is expected to be a tuple: (x_missing, x_complete)
        x_miss, x_true = x
        self.features_acquired_this_step = 0
        x_acquired = x_miss.copy()

        # Identify missing features
        missing_features = [f for f, v in x_miss.items() if v is None or pd.isna(v)]
        if not missing_features:
            return x_acquired

        # Get feature merits from the scorer
        merits = self.scorer.get_merits(missing_features, self.feature_costs)

        # Select a set of acquisition candidates based on strategy
        acquisition_candidates = self._select_candidates(x_miss, merits, missing_features)

        if not acquisition_candidates:
            return x_acquired

        # --- Batch Acquisition Logic ---
        # 1. Calculate cost and quality gain for the whole set of candidates
        total_cost = sum(self.feature_costs.get(f, 1) for f in acquisition_candidates)

        known_features_before = [f for f, v in x_miss.items() if v is not None and not pd.isna(v)]
        quality_before = self._calculate_quality(known_features_before, merits)

        hypothetical_features_after = known_features_before + acquisition_candidates
        quality_after = self._calculate_quality(hypothetical_features_after, merits)

        quality_gain = quality_after - quality_before

        # 2. Ask budget manager for permission for the entire set
        if self.budget_manager.can_acquire(quality_gain, total_cost):
            # 3. If approved, permanently acquire all features in the set
            for feature_to_acquire in acquisition_candidates:
                x_acquired[feature_to_acquire] = x_true[feature_to_acquire]
            self.features_acquired_this_step = len(acquisition_candidates)

        return x_acquired

    def _select_candidates(self, x_miss, merits, missing_features):
        """Selects a set of feature acquisition candidates based on the chosen strategy."""

        # Filter merits to only include missing features
        missing_merits = {f: m for f, m in merits.items() if f in missing_features}
        sorted_merits = sorted(missing_merits.items(), key=lambda item: item[1], reverse=True)

        if self.acquisition_strategy == 'k-best':
            # Select up to k best missing features based on local merits
            return [f for f, m in sorted_merits][:self.k]

        elif self.acquisition_strategy == 'k-global-best':
            # Select up to k best features that are also globally ranked
            global_merits = self.scorer.get_global_merits()
            sorted_global = sorted(global_merits.items(), key=lambda item: item[1], reverse=True)
            top_k_global_features = {f for f, m in sorted_global[:self.k]}

            candidates = []
            for f, m in sorted_merits:
                if f in top_k_global_features:
                    candidates.append(f)
            return candidates  # Already capped by local missing features

        elif self.acquisition_strategy == 'k-max-mean':
            # Greedily add features that improve instance quality (mean merit)
            candidates = []

            # Start with known features from the missing instance
            known_features_merits = [merits.get(f, 0) for f, v in x_miss.items() if
                                     v is not None and not pd.isna(v) and f in merits]

            for feature, merit in sorted_merits:
                if len(candidates) >= self.k:
                    break

                # Check if adding this feature would increase the mean
                current_mean = np.mean(known_features_merits) if known_features_merits else 0
                if merit > current_mean:
                    candidates.append(feature)
                    known_features_merits.append(merit)
            return candidates

        else:
            raise ValueError(f"Unknown acquisition strategy: {self.acquisition_strategy}")
