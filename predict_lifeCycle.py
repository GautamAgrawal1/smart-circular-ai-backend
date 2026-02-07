import joblib
import numpy as np

model = joblib.load("lifecycle_model.pkl")

# Example inputs
condition_score = 0.33
age_months = 24
usage_level = 2

X = np.array([[condition_score, age_months, usage_level]])

remaining_life = model.predict(X)[0]

print("Remaining life (months):", round(remaining_life, 2))
