import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

data_path = os.path.join("data", "master_air_quality_dataset.parquet")
model_dir = "models"

meta_path = os.path.join("data", "multi_horizon_metadata.json")
if not os.path.exists(meta_path):
    meta_path = os.path.join(model_dir, "multi_horizon_metadata.json")

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Metadata file not found at {meta_path}. Run model training first!")

pred_dir = os.path.join("data", "predictions")
os.makedirs(pred_dir, exist_ok=True)

print("--- STABILIZED MULTI-HORIZON INFERENCE ENGINE ---")

with open(meta_path, "r") as f:
    meta = json.load(f)

# Safely parse top-level or nested feature columns
if "feature_cols" in meta:
    feature_cols = meta["feature_cols"]
else:
    # Find first horizon dictionary entry
    horizon_key = next((k for k in meta if isinstance(meta[k], dict) and "feature_cols" in meta[k]), None)
    if horizon_key:
        feature_cols = meta[horizon_key]["feature_cols"]
    else:
        raise ValueError("Could not extract 'feature_cols' from multi_horizon_metadata.json!")

df = pd.read_parquet(data_path).sort_values("date").reset_index(drop=True)

# Re-engineer required feature transformations
df["target_aqi_log"] = np.log1p(df["target_aqi"])
df["target_log"] = np.log1p(df["target_aqi"])

if "wind_speed_m_s" not in df.columns and "wind_u_10m" in df.columns:
    df["wind_speed_m_s"] = np.sqrt(df["wind_u_10m"]**2 + df["wind_v_10m"]**2)

if "wind_dir_rad" not in df.columns and "wind_u_10m" in df.columns:
    df["wind_dir_rad"] = np.arctan2(df["wind_v_10m"], df["wind_u_10m"])

df["delta_lag1"] = df["target_log"] - df["target_log"].shift(1)
df["delta_lag2"] = df["target_log"].shift(1) - df["target_log"].shift(2)
df["delta_lag3"] = df["target_log"].shift(2) - df["target_log"].shift(3)

for i in [1, 2, 3, 5, 7, 14]:
    df[f"target_aqi_log_lag{i}"] = df["target_aqi_log"].shift(i)

df_clean = df.dropna().reset_index(drop=True)
last_row = df_clean.iloc[-1]

if "date" in last_row:
    anchor_date = pd.to_datetime(last_row["date"])
else:
    anchor_date = pd.to_datetime(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

# Build anchor feature matrix
X_dict = {col: float(last_row[col]) if col in last_row else 0.0 for col in feature_cols}
X_anchor = pd.DataFrame([X_dict]).astype(np.float32)

execution_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
records = []

horizon_targets = [
    ("H1_24h", 1, ["H1_24h", "H1", "24h", "lead1", "1"]),
    ("H2_48h", 2, ["H2_48h", "H2", "48h", "lead2", "2"]),
    ("H3_72h", 3, ["H3_72h", "H3", "72h", "lead3", "3"])
]

for display_name, step_days, alias_keys in horizon_targets:
    h_info = None
    for key in alias_keys:
        if key in meta and isinstance(meta[key], dict):
            h_info = meta[key]
            break

    if h_info is None:
        print(f"[WARNING] Skipping {display_name}: Horizon missing in metadata JSON.")
        continue

    model_file = h_info.get("model_path") or h_info.get("models", {}).get("delta_model")
    if not model_file or not os.path.exists(model_file):
        print(f"[WARNING] Model file for {display_name} not found at: {model_file}")
        continue

    model = joblib.load(model_file)
    raw_pred_delta = float(model.predict(X_anchor)[0])
    
    current_log_aqi = float(last_row["target_aqi_log"])
    pred_val = float(np.clip(np.expm1(current_log_aqi + raw_pred_delta), 0.0, 500.0))

    # Error interval boundaries
    q05 = h_info.get("q05_margin", -0.15)
    q95 = h_info.get("q95_margin", 0.15)
    
    lower_bound = float(np.clip(np.expm1(current_log_aqi + raw_pred_delta + q05), 0.0, 500.0))
    upper_bound = float(np.clip(np.expm1(current_log_aqi + raw_pred_delta + q95), 0.0, 500.0))

    forecast_date = anchor_date + pd.Timedelta(days=step_days)

    if pred_val <= 50: cat = "Good"
    elif pred_val <= 100: cat = "Moderate"
    elif pred_val <= 150: cat = "Unhealthy for Sensitive Groups"
    elif pred_val <= 200: cat = "Unhealthy"
    elif pred_val <= 300: cat = "Very Unhealthy"
    else: cat = "Hazardous"

    r2_score = h_info.get("honest_cv_r2_mean", h_info.get("cv_r2_mean", 0.0))

    records.append({
        "execution_timestamp": execution_ts,
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "forecast_horizon": display_name,
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "predicted_aqi": round(pred_val, 2),
        "ci_95_lower": round(lower_bound, 2),
        "ci_95_upper": round(upper_bound, 2),
        "aqi_category": cat,
        "cv_r2_score": round(r2_score, 3)
    })

if not records:
    raise RuntimeError("Engine failed to generate predictions. Verify metadata keys.")

df_pred = pd.DataFrame(records)

# Execution Persistence
timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
run_parquet_path = os.path.join(pred_dir, f"forecast_run_{timestamp_str}.parquet")
df_pred.to_parquet(run_parquet_path, engine="pyarrow", index=False)

history_parquet_path = os.path.join(pred_dir, "forecast_history.parquet")
if os.path.exists(history_parquet_path):
    df_hist = pd.read_parquet(history_parquet_path)
    df_combined = pd.concat([df_hist, df_pred], ignore_index=True)
    df_combined.to_parquet(history_parquet_path, engine="pyarrow", index=False)
else:
    df_pred.to_parquet(history_parquet_path, engine="pyarrow", index=False)

print("\n--- OFFICIAL 72-HOUR AQI FORECAST ---")
print(df_pred[["forecast_horizon", "forecast_date", "predicted_aqi", "ci_95_lower", "ci_95_upper", "aqi_category", "cv_r2_score"]].to_string(index=False))
print(f"\nSaved execution artifact to: {run_parquet_path}")