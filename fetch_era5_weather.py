import os
import ssl
import geopandas as gpd
import pandas as pd
import urllib3
import ee

# Bypass local SSL context issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ID = "green-cover-karachi"
ee.Initialize(project=PROJECT_ID)

# Load local Karachi boundary
geojson_path = os.path.join("data", "karachi_only.geojson")
gdf = gpd.read_file(geojson_path)
min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
karachi_geometry = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

years = [2023, 2024, 2025, 2026]
all_dfs = []

print("--- EXPORTING ERA5-LAND METEOROLOGICAL DATA TO PARQUET ---")

for year in years:
    start_str = f"{year}-01-01"
    end_str = f"{year}-12-31" if year < 2026 else "2026-08-31"
    
    print(f"Processing weather data for year: {year}...")
    
    start_date = ee.Date(start_str)
    end_date = ee.Date(end_str)
    n_days = end_date.difference(start_date, 'day')
    day_list = ee.List.sequence(0, n_days.subtract(1))
    
    # ERA5-Land Daily Aggregated Dataset
    era5_coll = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .select([
            "temperature_2m", 
            "surface_pressure", 
            "u_component_of_wind_10m", 
            "v_component_of_wind_10m",
            "total_precipitation_sum"
        ])
        .filterBounds(karachi_geometry)
        .filterDate(start_str, end_str)
    )

    def process_day(day_offset):
        current_date = start_date.advance(day_offset, 'day')
        next_date = current_date.advance(1, 'day')
        
        filtered = era5_coll.filterDate(current_date, next_date)
        has_images = filtered.size().gt(0)
        
        def calc_stats():
            daily_img = filtered.mean()
            stats = daily_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=karachi_geometry,
                scale=10000, # ERA5-Land resolution is ~10km
                maxPixels=1e9
            )
            return ee.Feature(None, {
                "date": current_date.format("YYYY-MM-dd"),
                "temp_2m_k": stats.get("temperature_2m"),
                "surface_pressure_pa": stats.get("surface_pressure"),
                "wind_u_10m": stats.get("u_component_of_wind_10m"),
                "wind_v_10m": stats.get("v_component_of_wind_10m"),
                "precipitation_m": stats.get("total_precipitation_sum")
            })
            
        def empty_stats():
            return ee.Feature(None, {"date": current_date.format("YYYY-MM-dd"), "temp_2m_k": None})

        return ee.Feature(ee.Algorithms.If(has_images, calc_stats(), empty_stats()))

    daily_fc = ee.FeatureCollection(day_list.map(process_day))
    features = daily_fc.getInfo()["features"]
    
    data_list = [
        f["properties"] for f in features 
        if f.get("properties") and f["properties"].get("temp_2m_k") is not None
    ]
    if data_list:
        df_year = pd.DataFrame(data_list)
        all_dfs.append(df_year)

# Concatenate all years into a single DataFrame
df_era5 = pd.concat(all_dfs, ignore_index=True)
df_era5["date"] = pd.to_datetime(df_era5["date"])
df_era5 = df_era5.sort_values("date").reset_index(drop=True)

# Derived Feature Engineering: Convert Kelvin to Celsius & compute Wind Speed
df_era5["temp_2m_c"] = df_era5["temp_2m_k"] - 273.15
df_era5["wind_speed_m_s"] = (df_era5["wind_u_10m"]**2 + df_era5["wind_v_10m"]**2)**0.5

# Save directly to Apache Parquet
output_parquet = os.path.join("data", "era5_karachi_daily.parquet")
df_era5.to_parquet(output_parquet, engine="pyarrow", index=False)

print(f"\nSUCCESS: Exported {len(df_era5)} daily weather records to: {output_parquet}")
print("\nFirst 5 rows of ERA5 Weather Dataset:")
print(df_era5[["date", "temp_2m_c", "wind_speed_m_s", "surface_pressure_pa", "precipitation_m"]].head())