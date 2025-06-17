from .BudgetManager import BudgetManager

class SimpleBudgetManager(BudgetManager):
    """Acquires features as long as there is available budget."""

    def can_acquire(self, quality_gain: float, acquisition_cost: float) -> bool:
        if self.total_budget_spent + acquisition_cost <= self.total_budget_received:
            self.total_budget_spent += acquisition_cost
            return True
        return False