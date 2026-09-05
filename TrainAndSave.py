import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

def train_and_save():
    print("Loading data...")
    # 1. Load Data
    df = pd.read_csv("100m_splits_combined.csv")

    # 2. Define Features and Target
    feature_cols = [
        "reaction time", "wind",
        "time 10m", "time 20m", "time 30m", "time 40m", "time 50m",
        "time 60m", "time 70m", "time 80m", "time 90m",
        "velocity 10m", "velocity 20m", "velocity 30m", "velocity 40m",
        "velocity 50m", "velocity 60m", "velocity 70m", "velocity 80m",
        "velocity 90m"
    ]
    
    X = df[feature_cols].copy()
    y = df["time 100m"].copy()

    print("Cleaning and preprocessing data...")
    # 3. Clean 'wind' column
    X['wind'] = X['wind'].astype(str).str.replace(' m/s', '', regex=False)

    # Convert all feature columns to numeric and impute NaNs with the mean
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())

    # Convert target to numeric and impute NaNs
    y = pd.to_numeric(y, errors='coerce')
    if y.isnull().any():
        y = y.fillna(y.mean())

    print("Training the model...")
    # 4. Initialize and Train Model
    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42
    )
    
    model.fit(X, y)

    print("Saving the model...")
    # 5. Save the Model via Joblib
    joblib.dump(model, "100m_model.pkl")
    print("Success! Model saved to '100m_model.pkl'")

if __name__ == "__main__":
    train_and_save()