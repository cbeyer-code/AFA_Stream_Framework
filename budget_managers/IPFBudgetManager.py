import collections
import math
import numpy as np
from .BudgetManager import BudgetManager

class IPFBudgetManager(BudgetManager):
    """
    Incremental Percentile Filter (IPF) for dynamic budgeting.
    Adapts the purchase threshold based on spending behavior.
    """

    def __init__(self, budget_per_instance: float, window_size: int = 100, penalty_constant: float = 16.0):
        super().__init__(budget_per_instance)
        self.window_size = window_size
        self.penalty_constant = penalty_constant
        self.quality_gain_window = collections.deque(maxlen=window_size)
        self.recent_acquisition_costs = collections.deque(maxlen=window_size)
        self.threshold = 0.2  # Initial threshold

    def can_acquire(self, quality_gain: float, acquisition_cost: float) -> bool:
        self.quality_gain_window.append(quality_gain)

        # Check if quality gain is above the current threshold percentile
        if len(self.quality_gain_window) < 2:  # Need at least 2 points to compute percentile
            return False

        percentile_value = np.percentile(self.quality_gain_window, (1 - self.threshold) * 100)
        #print(f'percentile_value {percentile_value} quality_gain {quality_gain}')
        #print(self.quality_gain_window)
        if quality_gain >= percentile_value:
            self.total_budget_spent += acquisition_cost
            self.recent_acquisition_costs.append(acquisition_cost)
            return True

        return False

    def learn_one(self, *args, **kwargs):
        super().learn_one()
        self._update_threshold()

    def _update_threshold(self):
        """Adjusts the IPF threshold based on budget usage, as per the paper."""
        if not self.recent_acquisition_costs or self.total_budget_received == 0:
            return

        # Estimate average cost of recent acquisitions
        avg_cost_estimate = np.mean(self.recent_acquisition_costs)

        # Preliminary threshold (T_pre)
        t_pre = self.budget_per_instance / avg_cost_estimate if avg_cost_estimate > 0 else 0

        # Budget usage ratio
        b_used = self.total_budget_spent / self.total_budget_received if self.total_budget_received > 0 else 1.0

        # Basic threshold (T_basic)
        t_basic = t_pre / b_used if b_used > 0 else t_pre

        # Final threshold with penalty for overspending
        if b_used > 1.0:
            penalty = math.floor(self.penalty_constant * (b_used - 1)) + 2
            self.threshold = t_basic / penalty
        else:
            self.threshold = t_basic

        # Clamp threshold to a reasonable range [0, 1]
        self.threshold = max(0.0, min(1.0, self.threshold))
