import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "processed_fraud_train.csv"

def compare_models():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Select features
    features = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
    target = "is_fraud"
    
    X = df[features].copy()
    y = df[target]
    
    # Encode categorical features
    categorical_cols = ["category", "gender", "state", "merchant"]
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Standardize for Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_s, X_test_s, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    results = {}

    # 1. Random Forest
    print("\nEvaluating Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "F1 (Fraud)": f1_score(y_test, y_pred_rf)
    }
    print(classification_report(y_test, y_pred_rf))

    # 2. Logistic Regression
    print("\nEvaluating Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    results["Logistic Regression"] = {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "F1 (Fraud)": f1_score(y_test, y_pred_lr)
    }
    print(classification_report(y_test, y_pred_lr))

    print("\n--- Summary Comparison ---")
    summary_df = pd.DataFrame(results).T
    print(summary_df)
    
    return summary_df

if __name__ == "__main__":
    compare_models()
