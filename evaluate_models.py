import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/cleaned_data.csv")

X = data.drop(columns=["median_house_value"])
y = data["median_house_value"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)

model_paths = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost_tuned": "models/xgboost_best.pkl",
    "Decision Tree": "models/decision_tree.pkl",
     "XGBoost": "models/xgboost.pkl",
}

results = {}

for model_name, model_path in model_paths.items():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test_scaled)  # Use only test data
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {"RMSE": rmse, "R2 Score": r2}
    print(f"{model_name}: RMSE = {rmse:.2f}, R² Score = {r2:.2f}")

results_df = pd.DataFrame(results).T

results_df.to_csv("models/model_evaluation_results.csv", index=True)
print("Evaluation results saved successfully!")

# Plot RMSE and R² Score 
fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar plot for RMSE
ax1.bar(results_df.index, results_df["RMSE"], color='skyblue', label='RMSE')
ax1.set_ylabel("RMSE", color='blue')
ax1.set_xlabel("Model")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title("Model Evaluation: RMSE and R² Score")

# Line plot for R² Score
ax2 = ax1.twinx()
ax2.plot(results_df.index, results_df["R2 Score"], color='red', marker='o', label='R² Score')
ax2.set_ylabel("R² Score", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Save the plot 
plt.savefig("models/model_comparison_chart.png")
plt.show()

print("Model evaluation chart saved successfully!")
