import os
import pandas as pd
import numpy as np

data_path = os.path.join("data", "master_air_quality_dataset.parquet")

if not os.path.exists(data_path):
    print(f"Error: Could not find dataset at {data_path}")
    exit()

df = pd.read_parquet(data_path)

print("=== DATASET OVERVIEW ===")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print("\n=== COLUMN LIST ===")
for col in df.columns:
    print(f"- {col}")

print("\n=== TOP CORRELATIONS WITH TARGET (aqi) ===")
numeric_df = df.select_dtypes(include=[np.number])
if "target_aqi" in numeric_df.columns:
    corrs = numeric_df.corr()["target_aqi"].sort_values(ascending=False)
    print(corrs.head(15))
    print("\n--- Negative Correlations ---")
    print(corrs.tail(10))
elif "aqi" in numeric_df.columns:
    corrs = numeric_df.corr()["aqi"].sort_values(ascending=False)
    print(corrs.head(15))