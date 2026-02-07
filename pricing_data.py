import numpy as np
import pandas as pd

np.random.seed(42)
rows = []

for _ in range(300):
    condition_score = np.round(np.random.uniform(0.2, 1.0), 2)
    remaining_life = np.random.uniform(1, 36)     # months
    age_months = np.random.randint(1, 60)
    base_price = np.random.randint(5000, 30000)
    demand_level = np.random.choice([1, 2, 3])    # low, medium, high

    # Pricing logic (simple & explainable)
    price = (
        base_price
        * condition_score
        * (remaining_life / 36)
        * (1 + 0.1 * (demand_level - 2))
    )

    price = max(500, price)  # floor price
    rows.append([condition_score, remaining_life, age_months, base_price, demand_level, price])

df = pd.DataFrame(rows, columns=[
    "condition_score",
    "remaining_life_months",
    "age_months",
    "base_price",
    "demand_level",
    "recommended_price"
])

df.to_csv("pricing_data.csv", index=False)
print("Pricing dataset created")
