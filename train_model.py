import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

print("--- TRAINING MULTI-HORIZON DELTA MODELS (H1, H2, H3) ---")

data_path = os.path.join("data", "master_air_quality_dataset.parquet")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

df = pd.read_parquet(data_path).sort_values("date").reset_index(drop=True)

# 1. STATIONARY TARGET & MOMENTUM TRANSFORMATIONS
df["target_aqi_log"] = np.log1p(df["target_aqi"])
df["target_log"] = df["target_aqi_log"]

if "wind_speed_m_s" not in df.columns and "wind_u_10m" in df.columns:
    df["wind_speed_m_s"] = np.sqrt(df["wind_u_10m"]**2 + df["wind_v_10m"]**2)

if "wind_dir_rad" not in df.columns and "wind_u_10m" in df.columns:
    df["wind_dir_rad"] = np.arctan2(df["wind_v_10m"], df["wind_u_10m"])

# Difference features
df["delta_lag1"] = df["target_log"] - df["target_log"].shift(1)
df["delta_lag2"] = df["target_log"].shift(1) - df["target_log"].shift(2)
df["delta_lag3"] = df["target_log"].shift(2) - df["target_log"].shift(3)

for i in [1, 2, 3, 5, 7, 14]:
    df[f"target_aqi_log_lag{i}"] = df["target_aqi_log"].shift(i)

# Feature selection
exclude_cols = ["date", "target_aqi", "target_aqi_log", "target_log"]
feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("target_lead")]

multi_metadata = {"feature_cols": feature_cols}

horizons = [
    ("H1_24h", 1),
    ("H2_48h", 2),
    ("H3_72h", 3)
]

tscv = TimeSeriesSplit(n_splits=5)

for h_name, lead_step in horizons:
    print(f"\n--- Training {h_name} (Lead Step: {lead_step}) ---")
    
    # Define stationary Delta target: log(AQI_{t+k}) - log(AQI_t)
    df_h = df.copy()
    df_h["target_lead_log"] = df_h["target_log"].shift(-lead_step)
    df_h["target_delta"] = df_h["target_lead_log"] - df_h["target_log"]
    
    df_clean = df_h.dropna(subset=feature_cols + ["target_delta"]).reset_index(drop=True)
    
    X = df_clean[feature_cols].values
    y_delta = df_clean["target_delta"].values
    y_actual_log = df_clean["target_lead_log"].values
    y_current_log = df_clean["target_log"].values
    
    cv_r2_scores = []
    cv_mae_scores = []
    residuals = []
    
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr_delta, y_val_delta = y_delta[train_idx], y_delta[val_idx]
        
        # Train Ridge model on Delta stationary targets
        model_cv = Ridge(alpha=10.0)
        model_cv.fit(X_tr, y_tr_delta)
        
        # Predict Delta and reconstruct absolute Log AQI
        pred_delta = model_cv.predict(X_val)
        pred_log = y_current_log[val_idx] + pred_delta
        
        actual_log = y_actual_log[val_idx]
        
        # Evaluate on reconstructed log values
        cv_r2_scores.append(r2_score(actual_log, pred_log))
        cv_mae_scores.append(mean_absolute_error(actual_log, pred_log))
        residuals.extend(actual_log - pred_log)
    
    mean_r2 = np.mean(cv_r2_scores)
    mean_mae = np.mean(cv_mae_scores)
    print(f"Validation R2 Score: {mean_r2:.4f}")
    print(f"Validation MAE Score: {mean_mae:.4f}")
    
    # Calculate 95% Quantile Margin for Confidence Intervals
    q05 = float(np.quantile(residuals, 0.05))
    q95 = float(np.quantile(residuals, 0.95))
    
    # Train final model on full dataset
    final_model = Ridge(alpha=10.0)
    final_model.fit(X, y_delta)
    
    model_path = os.path.join(model_dir, f"ridge_delta_{h_name.lower()}.pkl")
    joblib.dump(final_model, model_path)
    
    multi_metadata[h_name] = {
        "model_path": model_path,
        "target_col": f"target_delta_lead{lead_step}",
        "honest_cv_r2_mean": float(mean_r2),
        "honest_cv_mae_mean": float(mean_mae),
        "q05_margin": q05,
        "q95_margin": q95
    }

# Save structured multi-horizon metadata JSON
meta_out_path = os.path.join(model_dir, "multi_horizon_metadata.json")
with open(meta_out_path, "w") as f:
    json.dump(multi_metadata, f, indent=4)

print(f"\nSaved multi-horizon metadata to: {meta_out_path}")