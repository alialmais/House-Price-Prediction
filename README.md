# House-Price-Prediction


**Overview:**
This project explores four machine learning models to determine the optimal method for predicting house prices in California. Linear regression was chosen first and considered as a baseline model due to it’s simplicity, decision tree was adopted to identify non-linear relationships in housing data, Random Forest was used to improve generalization, and XGBoost was selected due to it’s performance on structured data.
I used Root Mean Squared Error (RMSE) and R² Score for evaluating the models.
  RMSE helps in indicating   how much error the model makes. R² Score helps us know how well the model understands house prices. The model with the best performance was XGBoost(Tuned) achieving R² = 0.73.As future work I should focus on improving feature engineering to achieve higher accuracy.

---

## **Project Structure**

## Project Structure

- **house_price_prediction/**
  - `README.md` → Documentation
  - `dataCleaning.py`
  - **training/** → Trains machine learning models
    - `train_linear_regression.py`
    - `train_random_forest.py`
    - `train_xgboost.py`
    - `train_xgboost_tuned.py`
    - `train_decision_tree.py`
  - **data/**
    - `california_housing.csv`
    - `cleaned_data.csv`
  - **models/** → Stores trained models
    - `linear_regression.pkl`
    - `random_forest.pkl`
    - `xgboost_best.pkl`
    - `decision_tree.pkl`
    - `scaler.pkl`
  - `model_evaluation_results.csv`
  - `model_comparison_chart.png`
  - `evaluate_models.py` → Compares model performance
  - `test.py` → Loads & tests the best model
  - `main.py` → User inputs house details and gets a price prediction
  
##  **Dataset**
- **Source:** California Housing dataset.
- **Features Used:**
  - `longitude`, `latitude` 
  - `housing_median_age` 
  - `total_rooms`, `total_bedrooms` 
  - `population`, `households`
  - `median_income` 
  - `ocean_proximity` 



  ##  **Model Performance**
| Model                 | Training R² | Testing R² | Training RMSE | Testing RMSE |
|---------------------  |------------ |------------|---------------|--------------|
| **Linear Regression** | 0.62        | 0.60       |                              |
| **Random Forest**     | 0.79        | 0.69       |
| **XGBoost (Tuned)**   | 0.80        | 0.73       | 50572.77      | 58138.12     |
| **Decision Tree**     | 0.76        | 0.61       |                              |
 



