import random
import pandas as pd
import numpy as np

# Set random seed for reproducibility
rng = np.random.default_rng(seed=42)

# Create a synthetic stream with 10,000 instances
n_instances = 10_000
data = []

for i in range(n_instances):
    # Create features
    x = rng.random()
    y = rng.random()

    # Create categorical features
    color = random.choice(['red', 'blue', 'green'])
    shape = random.choice(['circle', 'square', 'triangle'])

    # Generate labels
    if i < 5000:
        # First half: numerical feature 'x' is highly correlated with label
        label = int(x > 0.5)
    else:
        # Second half: categorical feature 'color' determines label
        label = 1 if color == 'red' else 0

    data.append({
        'x': x,
        'y': y,
        'color': color,
        'shape': shape,
        'label': label
    })

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv('mixed_5k_shift')
# Automatically detect types from the first row
x_sample = df.drop(columns='label').iloc[0].to_dict()
numerical_features = [k for k, v in x_sample.items() if isinstance(v, (int, float))]
categorical_features = [k for k, v in x_sample.items() if not isinstance(v, (int, float))]

# Print the detected feature types
print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

