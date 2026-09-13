
import os
import shutil
import pandas as pd
 
DATA_PATH = os.path.join("data", "master_air_quality_dataset.parquet")
BACKUP_PATH = os.path.join("data", "master_air_quality_dataset.PRECLEAN.parquet")
 
# Confirmed leaked / orphaned legacy columns.
COLUMNS_TO_DROP = [
    "target_res_lead1",   # direct leakage: corr 0.63 with future target
    "target_res_lead2",   # direct leakage
    "target_res_lead3",   # direct leakage
    "target_residual_log",
    "target_residual_log_lag1",
    "target_residual_log_lag2",
    "target_residual_log_lag3",
    "target_residual_log_lag7",
    "aqi_baseline_log",
    "res_roll3_avg",
    "res_roll7_avg",
    "res_roll7_std",
]
 
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")
 
    shutil.copy2(DATA_PATH, BACKUP_PATH)
    print(f"Backup written to: {BACKUP_PATH}")
 
    df = pd.read_parquet(DATA_PATH)
    present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    missing = [c for c in COLUMNS_TO_DROP if c not in df.columns]
 
    print(f"Dropping {len(present)} stale/leaky columns: {present}")
    if missing:
        print(f"(Already absent, skipping: {missing})")
 
    df_clean = df.drop(columns=present)
    df_clean.to_parquet(DATA_PATH, engine="pyarrow", index=False)
    print(f"Cleaned dataset saved. Shape: {df.shape} -> {df_clean.shape}")
 
if __name__ == "__main__":
    main()
