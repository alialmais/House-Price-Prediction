import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

model_path = "models/xgboost_best.pkl"  
with open(model_path, "rb") as f:
    model = pickle.load(f)

scaler_path = "models/scaler.pkl"
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

data = pd.read_csv("data/cleaned_data.csv")

new_data = pd.DataFrame({
    "longitude": [-122.23],
    "latitude": [37.88],
    "housing_median_age": [15],
    "total_rooms": [3000],
    "total_bedrooms": [500],
    "population": [1500],
    "households": [450],
    "median_income": [6.5],
    "ocean_proximity_INLAND": [1],
    "ocean_proximity_ISLAND": [0],
    "ocean_proximity_NEAR BAY": [0],
    "ocean_proximity_NEAR OCEAN": [0]
})

features = ["longitude", "latitude", "housing_median_age", "total_rooms", 
            "total_bedrooms", "population", "households", "median_income", 
            "ocean_proximity_INLAND", "ocean_proximity_ISLAND", 
            "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN"]

scaled_data = scaler.transform(data[features])
scaled_new_data = scaler.transform(new_data[features])
distances = euclidean_distances(scaled_data, scaled_new_data)

similar_indices = np.argsort(distances, axis=0)[:30].flatten()
similar_houses = data.iloc[similar_indices]

similar_price = similar_houses["median_house_value"].median()
print(f"Average price for the 30 most similar houses: ${similar_price:,.2f}")

new_data_scaled = scaler.transform(new_data)


predicted_price = model.predict(new_data_scaled)

print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
