import abc
from river import base

class BudgetManager(base.Base, abc.ABC):
    """Abstract base class for budget management methods."""

    def __init__(self, budget_per_instance: float):
        self.budget_per_instance = budget_per_instance
        self.total_budget_received = 0.0
        self.total_budget_spent = 0.0

    @abc.abstractmethod
    def can_acquire(self, quality_gain: float, acquisition_cost: float) -> bool:
        """Decides if a feature set can be acquired."""
        pass

    def get_spent_budget(self) -> float:
        return self.total_budget_spent

    def get_received_budget(self) -> float:
        return self.total_budget_received

    def learn_one(self, *args, **kwargs):
        """Update budget received for one instance."""
        self.total_budget_received += self.budget_per_instance