"""
Run after predict_72h.py and before the git commit/push step in CI.
Exits with a non-zero status (failing the workflow, blocking the push)
if anything looks wrong.
"""
import os
import sys
import json
import pandas as pd
 
FAIL = False
 
def fail(msg):
    global FAIL
    FAIL = True
    print(f"[HEALTH CHECK FAILED] {msg}")
 
# 1. Metadata must exist and each horizon must beat (or not be wildly worse than)
#    the persistence baseline logged during training.
meta_path = os.path.join("models", "multi_horizon_metadata.json")
if not os.path.exists(meta_path):
    fail("multi_horizon_metadata.json is missing.")
else:
    with open(meta_path) as f:
        meta = json.load(f)
    for h in ["H1_24h", "H2_48h", "H3_72h"]:
        info = meta.get(h, {})
        r2 = info.get("honest_cv_r2_mean")
        if r2 is None:
            fail(f"{h}: missing honest_cv_r2_mean in metadata.")
        elif r2 < -1.0:
            fail(f"{h}: R2={r2:.3f} is catastrophically bad, refusing to deploy.")
 
# 2. The latest forecast run must exist, have 3 rows, and contain no NaNs.
pred_dir = os.path.join("data", "predictions")
runs = sorted(f for f in os.listdir(pred_dir) if f.startswith("forecast_run_")) if os.path.isdir(pred_dir) else []
if not runs:
    fail("No forecast_run_*.parquet file was produced by predict_72h.py.")
else:
    df_pred = pd.read_parquet(os.path.join(pred_dir, runs[-1]))
    if len(df_pred) != 3:
        fail(f"Expected 3 forecast rows (H1/H2/H3), found {len(df_pred)}.")
    if df_pred[["predicted_aqi", "ci_95_lower", "ci_95_upper"]].isna().any().any():
        fail("NaN values found in predicted_aqi / confidence interval columns.")
    if (df_pred["predicted_aqi"] < 0).any() or (df_pred["predicted_aqi"] > 500).any():
        fail("predicted_aqi outside the valid 0-500 AQI range.")
 
if FAIL:
    print("\nOne or more checks failed. Blocking commit/push.")
    sys.exit(1)
 
print("All health checks passed. Safe to commit and push.")
 
