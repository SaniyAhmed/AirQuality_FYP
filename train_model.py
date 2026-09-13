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
BASE_EXCLUDE = ["date", "target_aqi", "target_aqi_log", "target_log"]

def is_leak_risk(col_name: str) -> bool:
    """Anything that could plausibly contain forward-looking / target-derived
    information gets excluded from features, not just an exact-prefix match."""
    leaky_substrings = ["lead", "target_delta", "target_residual", "aqi_baseline", "res_roll"]
    return any(s in col_name for s in leaky_substrings)

feature_cols = [c for c in df.columns if c not in BASE_EXCLUDE and not is_leak_risk(c)]

removed = [c for c in df.columns if c not in feature_cols and c not in BASE_EXCLUDE]
if removed:
    print(f"[SAFETY FILTER] Excluded {len(removed)} leak-risk / legacy columns: {removed}")

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
    baseline_r2_scores = []
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
        
        # Persistence Baseline: Predict target_log_{t+k} = target_log_t (Delta = 0)
        baseline_pred_log = y_current_log[val_idx]
        
        # Evaluate on reconstructed log values
        cv_r2_scores.append(r2_score(actual_log, pred_log))
        cv_mae_scores.append(mean_absolute_error(actual_log, pred_log))
        baseline_r2_scores.append(r2_score(actual_log, baseline_pred_log))
        residuals.extend(actual_log - pred_log)
    
    mean_r2 = np.mean(cv_r2_scores)
    mean_mae = np.mean(cv_mae_scores)
    mean_baseline_r2 = np.mean(baseline_r2_scores)
    
    print(f"Validation R2 Score: {mean_r2:.4f}")
    print(f"Validation MAE Score: {mean_mae:.4f}")
    print(f"Persistence Baseline R2 Score: {mean_baseline_r2:.4f}")
    
    if mean_r2 <= mean_baseline_r2:
        print(f"WARNING: Model {h_name} does not outperform persistence baseline!")
    
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
        "persistence_baseline_r2": float(mean_baseline_r2),
        "q05_margin": q05,
        "q95_margin": q95
    }

# Save structured multi-horizon metadata JSON
meta_out_path = os.path.join(model_dir, "multi_horizon_metadata.json")
with open(meta_out_path, "w") as f:
    json.dump(multi_metadata, f, indent=4)

print(f"\nSaved multi-horizon metadata to: {meta_out_path}")

# Export persistence baseline summary CSV for academic reporting
baseline_summary = pd.DataFrame([
    {
        "horizon": h,
        "model_r2": multi_metadata[h]["honest_cv_r2_mean"],
        "persistence_r2": multi_metadata[h]["persistence_baseline_r2"],
        "beats_baseline": multi_metadata[h]["honest_cv_r2_mean"] > multi_metadata[h]["persistence_baseline_r2"]
    }
    for h in ["H1_24h", "H2_48h", "H3_72h"]
])
baseline_summary.to_csv(os.path.join(model_dir, "baseline_comparison.csv"), index=False)
print("\nSaved baseline comparison to models/baseline_comparison.csv")
print(baseline_summary.to_string(index=False))