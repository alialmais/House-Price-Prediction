# House-Price-Prediction

![image](https://github.com/user-attachments/assets/c6fdf1aa-8f38-4d67-9e9d-304359f1d174)

---

## **Project Structure**


- **house_price_prediction/**
  - `README.md` 
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






