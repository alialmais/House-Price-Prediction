import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open("models/xgboost_best.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define feature names (matching the training dataset)
feature_names = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                 "total_bedrooms", "population", "households", "median_income", 
                 "ocean_proximity_INLAND", "ocean_proximity_ISLAND", 
                 "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN"]

# Get user input for house features
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

# Ask for categorical feature as a selection
ocean_options = ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
print("Choose Ocean Proximity:")
for i, option in enumerate(ocean_options):
    print(f"{i + 1}. {option}")

choice = int(input("Enter choice (1-4): ")) - 1

# Initialize ocean proximity features as 0
ocean_features = [0, 0, 0, 0]  # Corresponds to INLAND, ISLAND, NEAR BAY, NEAR OCEAN

# Set the chosen ocean proximity feature to 1
if 0 <= choice < len(ocean_features):
    ocean_features[choice] = 1

# Add ocean proximity values to user input
user_input.extend(ocean_features)

# Convert to DataFrame to retain feature names
input_df = pd.DataFrame([user_input], columns=feature_names)

# Apply feature scaling
scaled_input = scaler.transform(input_df)

# Make prediction
predicted_price = model.predict(scaled_input)[0]

# Print the result
print(f"\nPredicted House Price: ${predicted_price:,.2f}")
