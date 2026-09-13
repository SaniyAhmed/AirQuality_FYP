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

now_utc = datetime.now(timezone.utc)
start_fetch_date = (now_utc - timedelta(days=14)).strftime("%Y-%m-%d")
end_fetch_date = now_utc.strftime("%Y-%m-%d")

def _sub_index(c, breakpoints):
    """breakpoints: list of (C_low, C_high, I_low, I_high) tuples, EPA style."""
    if pd.isna(c) or c < 0:
        return np.nan
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= c <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo
    return breakpoints[-1][3]  # clamp to top category if above all ranges

# EPA breakpoint tables (concentration units as EPA defines them)
PM25_BP = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
           (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)]
PM10_BP  = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
            (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)]
CO_BP    = [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)]   # ppm, 8-hr
SO2_BP   = [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
            (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)]       # ppb, 1-hr
NO2_BP   = [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)]      # ppb, 1-hr
O3_BP    = [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
            (86, 105, 151, 200), (106, 200, 201, 300)]                              # ppb, 8-hr

def compute_epa_aqi(row):
    """Takes a row with pm2_5, pm10, carbon_monoxide, sulphur_dioxide,
    nitrogen_dioxide, ozone (Open-Meteo units: ug/m3 for particulates,
    ppb-equivalent handled below) and returns the official max-sub-index AQI."""
    sub_indices = []
    if "pm2_5" in row:
        sub_indices.append(_sub_index(row["pm2_5"], PM25_BP))
    if "pm10" in row:
        sub_indices.append(_sub_index(row["pm10"], PM10_BP))
    if "carbon_monoxide" in row:
        co_ppm = row["carbon_monoxide"] / 1145.0  # ug/m3 -> ppm approx conversion
        sub_indices.append(_sub_index(co_ppm, CO_BP))
    if "sulphur_dioxide" in row:
        so2_ppb = row["sulphur_dioxide"] / 2.62   # ug/m3 -> ppb approx conversion
        sub_indices.append(_sub_index(so2_ppb, SO2_BP))
    if "nitrogen_dioxide" in row:
        no2_ppb = row["nitrogen_dioxide"] / 1.88  # ug/m3 -> ppb approx conversion
        sub_indices.append(_sub_index(no2_ppb, NO2_BP))
    if "ozone" in row:
        o3_ppb = row["ozone"] / 1.96              # ug/m3 -> ppb approx conversion
        sub_indices.append(_sub_index(o3_ppb, O3_BP))

    sub_indices = [s for s in sub_indices if not pd.isna(s)]
    return max(sub_indices) if sub_indices else np.nan

aq_url = (f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LATITUDE}"
          f"&longitude={LONGITUDE}&start_date={start_fetch_date}&end_date={end_fetch_date}"
          f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=auto")
res_aq = requests.get(aq_url).json()
if "hourly" not in res_aq:
    raise RuntimeError(f"AQ API Error: {res_aq}")

df_aq = pd.DataFrame(res_aq["hourly"])
df_aq["date"] = pd.to_datetime(df_aq["time"]).dt.strftime("%Y-%m-%d")
df_daily_aq = df_aq.groupby("date").agg({
    "pm2_5": "mean", "pm10": "mean", "carbon_monoxide": "mean",
    "nitrogen_dioxide": "mean", "sulphur_dioxide": "mean", "ozone": "mean"
}).reset_index()
df_daily_aq["target_aqi"] = df_daily_aq.apply(compute_epa_aqi, axis=1)

wx_url = (f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}"
          f"&start_date={start_fetch_date}&end_date={end_fetch_date}"
          f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
          f"&timezone=auto")
res_wx = requests.get(wx_url).json()
if "hourly" not in res_wx:
    raise RuntimeError(f"Weather API Error: {res_wx}")

df_wx = pd.DataFrame(res_wx["hourly"])
df_wx["date"] = pd.to_datetime(df_wx["time"]).dt.strftime("%Y-%m-%d")
df_daily_wx = df_wx.groupby("date").agg({
    "temperature_2m": "mean", "relative_humidity_2m": "mean", "surface_pressure": "mean",
    "wind_speed_10m": "mean", "wind_direction_10m": "mean"
}).reset_index()

df_live = pd.merge(df_daily_aq, df_daily_wx, on="date", how="inner")

wind_rad_vals = np.radians(df_live["wind_direction_10m"])
df_live["wind_u_10m"] = -df_live["wind_speed_10m"] * np.sin(wind_rad_vals)
df_live["wind_v_10m"] = -df_live["wind_speed_10m"] * np.cos(wind_rad_vals)
df_live["wind_speed_m_s"] = df_live["wind_speed_10m"]
df_live["wind_dir_rad"] = wind_rad_vals

# ---------------------------------------------------------------------
# FIX #2 — MAP LIVE FIELDS ONTO THE MODEL'S TRAINING COLUMN NAMES.
# Previously these were left as NaN and silently frozen forever by ffill.
# ---------------------------------------------------------------------
df_live["temp_2m_c"] = df_live["temperature_2m"]
df_live["temp_2m_k"] = df_live["temperature_2m"] + 273.15
df_live["surface_pressure_pa"] = df_live["surface_pressure"] * 100.0  # hPa -> Pa
df_live["no2_mean"] = df_live["nitrogen_dioxide"]
df_live["no2_min"] = df_live["nitrogen_dioxide"]
df_live["no2_max"] = df_live["nitrogen_dioxide"]
# Open-Meteo's free forecast API does not provide AOD or precipitation-in-metres
# directly here; leave these as NaN for now (see Section 2, Step 2 for the
# proper satellite-AOD fix). Do NOT silently bfill them.
df_live["precipitation_m"] = np.nan

for col in df_existing.columns:
    if col not in df_live.columns:
        df_live[col] = np.nan
df_live = df_live[df_existing.columns]

df_combined = pd.concat([df_existing, df_live], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------
# FIX #2 — NO MORE bfill(). Forward-fill only (carry the last known value
# into a gap), never pull a future value backward into the past.
# A short remaining gap at the very start of the series (before ANY real
# observation exists) is left as NaN and handled by dropna() downstream,
# exactly like every other missing-data case in your pipeline.
# ---------------------------------------------------------------------
df_combined = df_combined.ffill()

df_combined.to_parquet(master_path, engine="pyarrow", index=False)
print(f"Dataset updated cleanly. Latest date: {df_combined['date'].iloc[-1]}")