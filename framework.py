import abc
import collections
import math
import random
import numpy as np
import pandas as pd
from river import (
    base,
    compose,
    datasets,
    evaluate,
    metrics,
    preprocessing,
    stats,
    stream,
    tree, linear_model, optim, forest,
)
import matplotlib.pyplot as plt

from afa_scores import AEDScorer,FeatureScorer, RandomScorer
from afa_strategies import ActiveFeatureAcquisition
from budget_managers import BudgetManager, IPFBudgetManager, SimpleBudgetManager


# --- 1. DATA PREPARATION ---

def create_missingness(dataset: pd.DataFrame, target_col: str, missingness_rate: float = 0.5, random_state: int = 42):
    """
    Introduces missing values into a dataset and creates two parallel streams.

    Args:
        dataset (pd.DataFrame): The complete input DataFrame.
        target_col (str): The name of the target variable column.
        missingness_rate (float): The proportion of feature values to make missing (0.0 to 1.0).
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: A tuple containing two river stream generators:
               - stream_missing: Yields (features_with_missing_values, target).
               - stream_complete: Yields (complete_features, target).
    """
    rng = np.random.RandomState(random_state)
    features = dataset.drop(columns=target_col)
    target = dataset[target_col]

    # Create the version with missing data
    features_missing = features.copy()
    for col in features.columns:
        if features[col].dtype.kind in 'biufc':  # Check if numeric or complex
            n_missing = int(len(features) * missingness_rate)
            if n_missing > 0:
                missing_indices = rng.choice(features.index, size=n_missing, replace=False)
                features_missing.loc[missing_indices, col] = np.nan

    # Create river streams
    stream_missing = stream.iter_pandas(X=features_missing, y=target)
    stream_complete = stream.iter_pandas(X=features, y=target)

    return stream_missing, stream_complete


# --- 2. FEATURE COST GENERATION ---

def generate_feature_costs(feature_names: list, strategy: str = 'equal', base_cost: int = 1) -> dict:
    """
    Generates a dictionary of feature costs based on a given strategy.

    Args:
        feature_names (list): A list of the feature names.
        strategy (str): One of 'equal', 'increasing', or 'decreasing'.
        base_cost (int): The base cost unit.

    Returns:
        dict: A dictionary mapping feature names to their costs.
    """
    if strategy == 'equal':
        return {name: base_cost for name in feature_names}
    elif strategy == 'increasing':
        return {name: (i + 1) * base_cost for i, name in enumerate(feature_names)}
    elif strategy == 'decreasing':
        return {name: (len(feature_names) - i) * base_cost for i, name in enumerate(feature_names)}
    else:
        raise ValueError("Strategy must be one of 'equal', 'increasing', or 'decreasing'.")








# --- 7. SIMULATION AND EVALUATION ---

def run_simulation(
        dataset,
        target_name,
        classifier_model,
        scorer,
        budget_manager,
        cost_strategy='equal',
        acquisition_strategy='k-best',
        k=1,
        missingness_rate=0.5,
        n_samples=5000
):
    """Main function to run the AFA simulation."""

    # 1. Prepare Data and Costs
    # Use the full dataset from the generator
    data_gen = iter(dataset)
    first_x, first_y = next(data_gen)
    feature_names = list(first_x.keys())

    # Re-create a Pandas DataFrame for missingness creation
    df_list = [{'y': first_y, **first_x}]
    for i, (x, y) in enumerate(data_gen):
        if i + 1 >= n_samples:
            break
        df_list.append({'y': y, **x})
    data_sample = pd.DataFrame(df_list).rename(columns={'y': target_name})

    stream_miss, stream_true = create_missingness(data_sample, target_name, missingness_rate)

    # Combine streams to pass (x_miss, x_true) tuples
    parallel_stream = zip(stream_miss, stream_true)

    # Generate costs
    costs = generate_feature_costs(feature_names, strategy=cost_strategy)

    # 2. Build Pipeline
    afa_transformer = ActiveFeatureAcquisition(
        scorer=scorer,
        budget_manager=budget_manager,
        feature_costs=costs,
        acquisition_strategy=acquisition_strategy,
        k=k
    )

    model_pipeline = compose.Pipeline(
        afa_transformer,
        preprocessing.StatImputer(),  # Impute any remaining missing values
        classifier_model
    )

    imputer = preprocessing.StatImputer()
    classifier = classifier_model

    # 3. Setup Metrics
    metric_kappa = metrics.CohenKappa()
    metric_accuracy = metrics.Accuracy()

    # History tracking
    history = {
        'step': [],
        'kappa': [],
        'accuracy': [],
        'budget_spent': [],
        'budget_received': [],
        'features_acquired': []
    }

    # 4. Run Progressive Validation
    print(
        f"Running simulation with: {acquisition_strategy}(k={k}), {scorer.__class__.__name__}, {budget_manager.__class__.__name__}, cost: {cost_strategy}")

    step = 0
    # Custom loop to handle the parallel stream and metric tracking
    for (x_miss, y_miss), (x_true, y_true) in parallel_stream:
        # --- PREDICTION FLOW ---
        # a. Perform Active Feature Acquisition
        x_acquired = afa_transformer.transform_one((x_miss, x_true))

        # b. Impute remaining missing values
        x_imputed = imputer.transform_one(x_acquired)

        # c. Make a prediction
        y_pred = classifier.predict_one(x_imputed)

        # --- METRIC UPDATE ---
        if y_pred is not None:
            metric_kappa.update(y_true, y_pred)
            metric_accuracy.update(y_true, y_pred)

        # --- LEARNING FLOW ---
        # a. Learn the AFA components (using true data)
        afa_transformer.learn_one((x_miss, x_true), y_true)

        # b. Learn the imputer (using the partially acquired data)
        imputer.learn_one(x_acquired)

        # c. Learn the classifier (using the fully imputed data)
        classifier.learn_one(x_imputed, y_true)

        # Record history
        if step % 10 == 0:
            history['step'].append(step)
            history['kappa'].append(metric_kappa.get())
            history['accuracy'].append(metric_accuracy.get())
            history['budget_spent'].append(budget_manager.get_spent_budget())
            history['budget_received'].append(budget_manager.get_received_budget())
            history['features_acquired'].append(afa_transformer.features_acquired_this_step)

        step += 1
        if step >= n_samples:
            break

    print("Simulation finished.")
    return history


def plot_results(history: dict, title: str):
    """Plots the results of the simulation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Performance (Kappa)
    ax1.plot(history['step'], history['kappa'], label='Kappa Score', color='blue')
    ax1.set_ylabel('Kappa Score')
    ax1.set_title(f'Model Performance - {title}')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(-1, 1)

    # Plot 2: Budget Usage
    ax2.plot(history['step'], history['budget_received'], label='Total Budget Received', linestyle='--', color='green')
    ax2.plot(history['step'], history['budget_spent'], label='Total Budget Spent', color='red')
    ax2.set_xlabel('Instances')
    ax2.set_ylabel('Budget')
    ax2.set_title('Budget Usage Over Time')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- CONFIGURATION ---
    N_SAMPLES = 20000
    BUDGET_PER_INSTANCE = 1.0
    MISSINGNESS = 0.5
    K_PARAM = 4  # For k-best, etc.

    # Select a dataset from river
    RIVER_DATASET = datasets.Elec2()
    #print(RIVER_DATASET.desc)
    TARGET = 'class'

    # --- CHOOSE YOUR COMPONENTS ---

    # 1. Classifier
    #classifier = tree.HoeffdingTreeClassifier(grace_period=100)
    #classifier = tree.ExtremelyFastDecisionTreeClassifier(grace_period=100,delta=1e-5, min_samples_reevaluate=100)
    #classifier = linear_model.LogisticRegression(optimizer=optim.SGD(.1))
    classifier = forest.AMFClassifier(n_estimators=10, use_aggregation=True, dirichlet=0.5, seed=1)

    # 2. Scorer
    #scorer = RandomScorer(seed=42)
    scorer = AEDScorer(window_size=200)

    # 3. Budget Manager
    #budget_mgr = SimpleBudgetManager(budget_per_instance=BUDGET_PER_INSTANCE)
    budget_mgr = IPFBudgetManager(budget_per_instance=BUDGET_PER_INSTANCE, window_size=200)

    # 4. Cost Strategy
    cost_strat = 'equal'  # 'equal', 'increasing', 'decreasing'

    # 5. Acquisition Strategy
    acq_strat = 'k-best'  # 'k-best', 'k-global-best', 'k-max-mean'

    # --- RUN ---

    history = run_simulation(
        dataset=RIVER_DATASET,
        target_name=TARGET,
        classifier_model=classifier,
        scorer=scorer,
        budget_manager=budget_mgr,
        cost_strategy=cost_strat,
        acquisition_strategy=acq_strat,
        k=K_PARAM,
        missingness_rate=MISSINGNESS,
        n_samples=N_SAMPLES
    )

    # --- VISUALIZE ---
    plot_title = (
        f"{acq_strat}(k={K_PARAM}), {scorer.__class__.__name__}, "
        f"{budget_mgr.__class__.__name__}\n"
        f"Cost: {cost_strat}, Budget/Inst: {BUDGET_PER_INSTANCE}, Missingness: {MISSINGNESS}"
    )
    plot_results(history, plot_title)
