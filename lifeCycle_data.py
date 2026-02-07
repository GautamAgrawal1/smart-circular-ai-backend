import numpy as np
import pandas as pd

np.random.seed(42)

data = []

for _ in range(200):
    condition_score = np.round(np.random.uniform(0.2, 1.0), 2)
    age_months = np.random.randint(1, 60)
    usage_level = np.random.choice([1, 2, 3])

    # Simple degradation logic
    base_life = 60  # months
    remaining_life = (
        base_life * condition_score
        - (age_months * 0.5)
        - (usage_level * 5)
    )

    remaining_life = max(0, remaining_life)

    data.append([
        condition_score,
        age_months,
        usage_level,
        remaining_life
    ])

df = pd.DataFrame(
    data,
    columns=[
        "condition_score",
        "age_months",
        "usage_level",
        "remaining_life_months"
    ]
)

df.to_csv("lifecycle_data.csv", index=False)
print("Lifecycle dataset created")
