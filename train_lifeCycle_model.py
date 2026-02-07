import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
df = pd.read_csv("lifecycle_data.csv")

X = df[["condition_score", "age_months", "usage_level"]]
y = df["remaining_life_months"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("Mean Absolute Error:", mae)

# Save model
joblib.dump(model, "lifecycle_model.pkl")
print("Lifecycle model saved")
