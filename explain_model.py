import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
 
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)
 
data_path = os.path.join("data", "master_air_quality_dataset.parquet")
meta_path = os.path.join("models", "multi_horizon_metadata.json")
 
print("--- SHAP EXPLAINABILITY: MATCHES THE DEPLOYED RIDGE MODELS ---")
 
with open(meta_path, "r") as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]
 
df = pd.read_parquet(data_path).sort_values("date").reset_index(drop=True)
df["target_aqi_log"] = np.log1p(df["target_aqi"])
df["target_log"] = df["target_aqi_log"]
if "wind_speed_m_s" not in df.columns and "wind_u_10m" in df.columns:
    df["wind_speed_m_s"] = np.sqrt(df["wind_u_10m"]**2 + df["wind_v_10m"]**2)
if "wind_dir_rad" not in df.columns and "wind_u_10m" in df.columns:
    df["wind_dir_rad"] = np.arctan2(df["wind_v_10m"], df["wind_u_10m"])
df["delta_lag1"] = df["target_log"] - df["target_log"].shift(1)
df["delta_lag2"] = df["target_log"].shift(1) - df["target_log"].shift(2)
df["delta_lag3"] = df["target_log"].shift(2) - df["target_log"].shift(3)
for i in [1, 2, 3, 5, 7, 14]:
    df[f"target_aqi_log_lag{i}"] = df["target_aqi_log"].shift(i)
 
df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
X = df_clean[feature_cols]
 
for h_name in ["H1_24h", "H2_48h", "H3_72h"]:
    model_path = meta[h_name]["model_path"]
    model = joblib.load(model_path)
    print(f"\nExplaining {h_name} using {model_path} ...")
 
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)
 
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"Karachi AQI Model ({h_name}): SHAP Beeswarm", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f"shap_beeswarm_{h_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()
 
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title(f"Global Feature Importance ({h_name})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f"shap_bar_{h_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()
 
    peak_idx = df_clean["target_aqi"].idxmax()
    peak_date = df_clean.loc[peak_idx, "date"]
    peak_aqi = df_clean.loc[peak_idx, "target_aqi"]
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[peak_idx], max_display=10, show=False)
    plt.title(f"Peak AQI Event Attribution ({h_name}): {peak_date} (AQI {peak_aqi:.1f})",
              fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f"shap_waterfall_{h_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()
 
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    df_ranking = pd.DataFrame({
        "Feature": feature_cols, "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    df_ranking.to_csv(os.path.join(reports_dir, f"shap_ranking_{h_name}.csv"), index=False)
 
    print(f"Top 5 drivers for {h_name}:")
    print(df_ranking.head(5).to_string(index=False))
 
print("\nDone. Each horizon now has its own SHAP report matching the model actually deployed for it.")
