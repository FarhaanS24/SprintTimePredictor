import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

def train_and_save_200m():
    print("Loading 200m data...")
    # Load Data
    df = pd.read_csv("200m_splits_combined.csv")

    # Define Features and Target
    feature_cols = [
        "reaction time", "wind",
        "time 50m", "time 100m", "time 150m",
        "velocity 50m", "velocity 100m", "velocity 150m"
    ]
    
    X = df[feature_cols].copy()
    y = df["time 200m"].copy()

    print("Cleaning and preprocessing data...")
    # Clean 'wind' column
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
    # Initialize and Train Model using hyperparameters from the 200m notebook
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=700,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42
    )
    
    model.fit(X, y)

    print("Saving the model...")
    # Save the Model via Joblib
    joblib.dump(model, "200m_model.pkl")
    print("Success! Model saved to '200m_model.pkl'")

if __name__ == "__main__":
    train_and_save_200m()