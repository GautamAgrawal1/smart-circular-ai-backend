import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("pricing_data.csv")

X = df[[
    "condition_score",
    "remaining_life_months",
    "age_months",
    "base_price",
    "demand_level"
]]
y = df["recommended_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("Pricing MAE:", round(mae, 2))

joblib.dump(model, "pricing_model.pkl")
print("Pricing model saved")
