import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Setup Output Directory
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

data_path = os.path.join("data", "master_air_quality_dataset.parquet")
model_path = os.path.join("models", "xgboost_aqi_model.joblib")

print("--- RUNNING INDUSTRY-GRADE SHAP EXPLAINABILITY ANALYSIS ---")

# 2. Load Model and Data
if not os.path.exists(model_path) or not os.path.exists(data_path):
    raise FileNotFoundError("Missing trained model or master dataset. Verify Phase 6 steps.")

model = joblib.load(model_path)
df = pd.read_parquet(data_path).sort_values("date").reset_index(drop=True)

# 3. Explicit Features Selection (Matching Training Feature Matrix)
explicit_features = [
    "no2_mean", "no2_min", "no2_max", "aod_047_mean",
    "temp_2m_c", "wind_speed_m_s", "wind_u_10m", "wind_v_10m", 
    "surface_pressure_pa", "precipitation_m", "sin_day", "cos_day",
    "est_pm25_lag1", "est_pm25_lag3", "est_pm25_lag7",
    "est_no2_ppb_lag1", "est_no2_ppb_lag3", "est_no2_ppb_lag7",
    "wind_speed_m_s_lag1", "wind_speed_m_s_lag3", "wind_speed_m_s_lag7",
    "target_aqi_lag1", "target_aqi_lag3", "target_aqi_lag7",
    "aqi_roll7_avg", "aqi_roll7_std"
]

feature_cols = [col for col in explicit_features if col in df.columns]
X = df[feature_cols]

# 4. Compute SHAP Values using TreeExplainer
print("Calculating TreeExplainer SHAP values across dataset...")
explainer = shap.TreeExplainer(model)
shap_explanation = explainer(X)

# 5. Plot 1: SHAP Beeswarm Plot (Feature Importance + Directional Impact)
print("Generating Plot 1: SHAP Summary Beeswarm...")
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
plt.title("Karachi AQI Model: SHAP Beeswarm Plot (Directional Feature Impact)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
beeswarm_path = os.path.join(reports_dir, "shap_summary_beeswarm.png")
plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
plt.close()

# 6. Plot 2: Global Mean |SHAP| Bar Chart
print("Generating Plot 2: SHAP Global Importance Bar Chart...")
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_explanation, max_display=15, show=False)
plt.title("Global Feature Importance (Mean Absolute SHAP Value)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
bar_path = os.path.join(reports_dir, "shap_feature_importance_bar.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

# 7. Plot 3: Local Peak-AQI Event Explanation (Waterfall Plot)
peak_idx = df["target_aqi"].idxmax()
peak_date = df.loc[peak_idx, "date"].strftime("%Y-%m-%d")
peak_aqi = df.loc[peak_idx, "target_aqi"]

print(f"Generating Plot 3: Local Waterfall Plot for Peak Pollution Day ({peak_date}, AQI: {peak_aqi:.1f})...")
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_explanation[peak_idx], max_display=10, show=False)
plt.title(f"Local Decision Attribution: Peak AQI Event on {peak_date} (True Target: {peak_aqi:.1f})", fontsize=11, fontweight="bold", pad=15)
plt.tight_layout()
waterfall_path = os.path.join(reports_dir, "shap_waterfall_peak_day.png")
plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
plt.close()

# 8. Export Numerical Impact Table
mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
df_ranking = pd.DataFrame({
    "Feature": feature_cols,
    "Mean_Absolute_SHAP": mean_abs_shap
}).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

csv_path = os.path.join(reports_dir, "shap_feature_impact_ranking.csv")
df_ranking.to_csv(csv_path, index=False)

print("\n--- SHAP ANALYSIS COMPLETE ---")
print(f" Saved Beeswarm Plot:         {beeswarm_path}")
print(f" Saved Global Bar Chart:       {bar_path}")
print(f" Saved Peak Event Waterfall:   {waterfall_path}")
print(f" Saved Numerical Impact CSV:   {csv_path}\n")

print("Top 10 Most Influential Features:")
print(df_ranking.head(10).to_string(index=False))