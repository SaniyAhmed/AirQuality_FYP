import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

LATITUDE = 24.8607
LONGITUDE = 67.0011

master_path = os.path.join("data", "master_air_quality_dataset.parquet")

print("--- FETCHING STABLE LIVE METEOROLOGICAL & AQI DATA FOR KARACHI ---")

if not os.path.exists(master_path):
    raise FileNotFoundError(f"Master dataset missing at {master_path}")

df_existing = pd.read_parquet(master_path)
df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.strftime("%Y-%m-%d")

# Define trailing lookback window (14 days through today)
now_utc = datetime.now(timezone.utc)
start_fetch_date = (now_utc - timedelta(days=14)).strftime("%Y-%m-%d")
end_fetch_date = now_utc.strftime("%Y-%m-%d")

# 1. Official US EPA Breakpoint Function for PM2.5
def pm25_to_aqi(pm):
    if pd.isna(pm) or pm < 0: return np.nan
    c = np.floor(10 * pm) / 10.0
    if c <= 12.0: return ((50 - 0) / (12.0 - 0.0)) * (c - 0.0) + 0
    elif c <= 35.4: return ((100 - 51) / (35.4 - 12.1)) * (c - 12.1) + 51
    elif c <= 55.4: return ((150 - 101) / (55.4 - 35.5)) * (c - 35.5) + 101
    elif c <= 150.4: return ((200 - 151) / (150.4 - 55.5)) * (c - 55.5) + 151
    elif c <= 250.4: return ((300 - 201) / (250.4 - 150.5)) * (c - 150.5) + 201
    elif c <= 500.4: return ((500 - 301) / (500.4 - 250.5)) * (c - 250.5) + 301
    else: return 500.0

# 2. Fetch Air Quality Observations
aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LATITUDE}&longitude={LONGITUDE}&start_date={start_fetch_date}&end_date={end_fetch_date}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=auto"
res_aq = requests.get(aq_url).json()

if "hourly" not in res_aq:
    raise RuntimeError(f"AQ API Error: {res_aq}")

df_aq = pd.DataFrame(res_aq["hourly"])
df_aq["date"] = pd.to_datetime(df_aq["time"]).dt.strftime("%Y-%m-%d")

df_daily_aq = df_aq.groupby("date").agg({
    "pm2_5": "mean",
    "pm10": "mean",
    "carbon_monoxide": "mean",
    "nitrogen_dioxide": "mean",
    "sulphur_dioxide": "mean",
    "ozone": "mean"
}).reset_index()

df_daily_aq["target_aqi"] = df_daily_aq["pm2_5"].apply(pm25_to_aqi)

# 3. Fetch Weather Data via Forecast API (Passing start_date and end_date only)
wx_url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&start_date={start_fetch_date}&end_date={end_fetch_date}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=auto"
res_wx = requests.get(wx_url).json()

if "hourly" not in res_wx:
    raise RuntimeError(f"Weather API Error: {res_wx}")

df_wx = pd.DataFrame(res_wx["hourly"])
df_wx["date"] = pd.to_datetime(df_wx["time"]).dt.strftime("%Y-%m-%d")

df_daily_wx = df_wx.groupby("date").agg({
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "surface_pressure": "mean",
    "wind_speed_10m": "mean",
    "wind_direction_10m": "mean"
}).reset_index()

# 4. Merge Observations & Engineer Vector Wind Components
df_live = pd.merge(df_daily_aq, df_daily_wx, on="date", how="inner")

wind_rad_vals = np.radians(df_live["wind_direction_10m"])
df_live["wind_u_10m"] = -df_live["wind_speed_10m"] * np.sin(wind_rad_vals)
df_live["wind_v_10m"] = -df_live["wind_speed_10m"] * np.cos(wind_rad_vals)
df_live["wind_speed_m_s"] = df_live["wind_speed_10m"]
df_live["wind_dir_rad"] = wind_rad_vals

for col in df_existing.columns:
    if col not in df_live.columns:
        df_live[col] = np.nan

df_live = df_live[df_existing.columns]

# 5. Concatenate & Deduplicate
df_combined = pd.concat([df_existing, df_live], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
df_combined = df_combined.ffill().bfill()

df_combined.to_parquet(master_path, engine="pyarrow", index=False)
print(f"Dataset updated cleanly. Latest date: {df_combined['date'].iloc[-1]}")