import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

if not os.path.exists("models"):
    os.makedirs("models")

data = pd.read_csv("data/cleaned_data.csv")

X = data.drop(columns=["median_house_value"])
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,  
    reg_alpha=1.0,  
    reg_lambda=1.0,  
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=True
)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {rmse_train:.2f}, Testing RMSE: {rmse_test:.2f}")
print(f"Training R²: {r2_train:.2f}, Testing R²: {r2_test:.2f}")

with open("models/xgboost_best.pkl", "wb") as f:
    pickle.dump(model, f)

print("Best XGBoost model saved successfully!")
