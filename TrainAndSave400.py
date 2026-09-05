import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

def train_and_save_400m():
    print("Loading 400m data...")
    # Load Data[cite: 5]
    df = pd.read_csv("400m_splits_combined.csv")

    # Define Features and Target[cite: 5]
    feature_cols = [
        "reaction time", "wind",
        "time 50m", "time 100m", "time 150m", "time 200m", "time 250m", "time 300m", "time 350m",
        "velocity 50m", "velocity 100m", "velocity 150m", "velocity 200m", "velocity 250m", "velocity 300m", "velocity 350m"
    ]
    
    X = df[feature_cols].copy()
    y = df["time 400m"].copy()

    print("Cleaning and preprocessing data...")
    # Clean 'wind' column if it exists as a string[cite: 5]
    if 'wind' in X.columns:
        X['wind'] = X['wind'].astype(str).str.replace(' m/s', '', regex=False)

    # Convert all feature columns to numeric and impute NaNs with the mean or 0[cite: 5]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isnull().any():
            mean_val = X[col].mean()
            fill_val = 0 if pd.isna(mean_val) else mean_val
            X[col] = X[col].fillna(fill_val)

    # Convert target to numeric and impute NaNs[cite: 5]
    y = pd.to_numeric(y, errors='coerce')
    if y.isnull().any():
        mean_val = y.mean()
        fill_val = 0 if pd.isna(mean_val) else mean_val
        y = y.fillna(fill_val)

    print("Training the model...")
    # Initialize and Train Model using hyperparameters from the 400m notebook[cite: 5]
    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42
    )
    
    model.fit(X, y)

    print("Saving the model...")
    # Save the Model via Joblib
    joblib.dump(model, "400m_model.pkl")
    print("Success! Model saved to '400m_model.pkl'")

if __name__ == "__main__":
    train_and_save_400m()