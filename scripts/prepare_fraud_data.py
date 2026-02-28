import pandas as pd
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "fraudTrain.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "processed_fraud_train.csv"

def prepare_data(sample_size=300000):
    print(f"Reading {INPUT_PATH}...")
    # Read specified columns only
    cols = [
        "trans_date_trans_time", "merchant", "category", "amt", 
        "first", "last", "gender", "state", "is_fraud"
    ]
    
    # We use a large chunk if possible, but for 335MB, reading once is fine on most systems.
    df = pd.read_csv(INPUT_PATH, usecols=cols)
    
    print(f"Original size: {len(df)}")
    
    # Balance the dataset (oversample fraud if necessary, or just sample)
    # Fraud is usually very small. Let's see the distribution.
    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0]
    
    print(f"Fraud count: {len(fraud_df)}")
    print(f"Legit count: {len(legit_df)}")
    
    # Oversample fraud to increase cases in database transition process
    fraud_df = pd.concat([fraud_df] * 5, ignore_index=True)
    print(f"Oversampled Fraud count: {len(fraud_df)}")
    
    # Sample to requested size while keeping as much oversampled fraud as possible
    n_fraud = len(fraud_df)
    n_legit = sample_size - n_fraud
    
    if n_legit > len(legit_df):
        n_legit = len(legit_df)
        
    sampled_legit = legit_df.sample(n_legit, random_state=42)
    final_df = pd.concat([fraud_df, sampled_legit]).sample(frac=1, random_state=42)
    
    print(f"Final sampled size: {len(final_df)}")
    
    # Basic preprocessing
    final_df["trans_date_trans_time"] = pd.to_datetime(final_df["trans_date_trans_time"])
    final_df["hour"] = final_df["trans_date_trans_time"].dt.hour
    final_df["day_of_week"] = final_df["trans_date_trans_time"].dt.dayofweek
    
    # Save processed data
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved processed data to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_data()
