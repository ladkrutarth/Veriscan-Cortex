import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import joblib
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "processed_fraud_train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model_rf.joblib"
ENCODERS_PATH = PROJECT_ROOT / "models" / "encoders.joblib"

def train_model():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Select features
    # requested: category, amt, first, last, gender, state, trans_date_trans_time, merchant
    # we added hour and day_of_week
    features = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
    target = "is_fraud"
    
    X = df[features].copy()
    y = df[target]
    
    # Encode categorical features
    encoders = {}
    categorical_cols = ["category", "gender", "state", "merchant"]
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(rf, MODEL_PATH)
    
    print(f"Saving encoders to {ENCODERS_PATH}...")
    joblib.dump(encoders, ENCODERS_PATH)
    
    print("Done!")

if __name__ == "__main__":
    train_model()
