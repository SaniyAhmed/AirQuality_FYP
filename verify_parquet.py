import os
import pandas as pd

file_path = os.path.join("data", "sentinel5p_karachi_daily.parquet")

# Load parquet file
df = pd.read_parquet(file_path)

print("--- PARQUET FILE AUDIT ---")
print(f"File Path: {file_path}")
print(f"File Size: {os.path.getsize(file_path) / 1024:.2f} KB")
print(f"Total Rows: {len(df)}")
print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print("\nColumn Data Types:")
print(df.dtypes)
print("\nNull Value Count:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())