import joblib
import numpy as np

model = joblib.load("pricing_model.pkl")

# Example inputs (plug outputs from Phase 1 & 2)
condition_score = 0.33
remaining_life_months = 8.4
age_months = 24
base_price = 20000
demand_level = 2  # medium

X = np.array([[condition_score, remaining_life_months, age_months, base_price, demand_level]])
price = model.predict(X)[0]

print("Recommended Price:", round(price, 2))
