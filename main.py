import pickle
import numpy as np
import pandas as pd

with open("models/xgboost_best.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_names = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                 "total_bedrooms", "population", "households", "median_income", 
                 "ocean_proximity_INLAND", "ocean_proximity_ISLAND", 
                 "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN"]

print("Enter house details:")

user_input = []
user_input.append(float(input("Longitude: ")))
user_input.append(float(input("Latitude: ")))
user_input.append(float(input("Housing Median Age: ")))
user_input.append(float(input("Total Rooms: ")))
user_input.append(float(input("Total Bedrooms: ")))
user_input.append(float(input("Population: ")))
user_input.append(float(input("Households: ")))
user_input.append(float(input("Median Income: ")))

ocean_options = ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
print("Choose Ocean Proximity:")
for i, option in enumerate(ocean_options):
    print(f"{i + 1}. {option}")

choice = int(input("Enter choice (1-4): ")) - 1

ocean_features = [0, 0, 0, 0] 

if 0 <= choice < len(ocean_features):
    ocean_features[choice] = 1

user_input.extend(ocean_features)

input_df = pd.DataFrame([user_input], columns=feature_names)

scaled_input = scaler.transform(input_df)

predicted_price = model.predict(scaled_input)[0]

print(f"\nPredicted House Price: ${predicted_price:,.2f}")
