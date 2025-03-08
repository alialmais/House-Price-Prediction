import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data/cleaned_data.csv")

X = data.drop(columns=["median_house_value"]) 
y = data["median_house_value"] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
r2 = r2_score(y_test, y_pred)  

print(f"Model Evaluation: RMSE = {rmse:.2f}, R² Score = {r2:.2f}")

with open("models/linear_regression.pkl", "wb") as f:
    pickle.dump(model, f)


with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Linear Regression model and scaler saved successfully!")
