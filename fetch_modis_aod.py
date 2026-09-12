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

print("--- EXPORTING MODIS MAIAC AOD (PARTICULATE PROXY) TO PARQUET ---")

for year in years:
    start_str = f"{year}-01-01"
    end_str = f"{year}-12-31" if year < 2026 else "2026-08-31"
    
    print(f"Processing MODIS AOD for year: {year}...")
    
    start_date = ee.Date(start_str)
    end_date = ee.Date(end_str)
    n_days = end_date.difference(start_date, 'day')
    day_list = ee.List.sequence(0, n_days.subtract(1))
    
    # NASA MODIS MAIAC 1km AOD Collection
    aod_coll = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .select("Optical_Depth_047") # 0.47 micron Blue band AOD
        .filterBounds(karachi_geometry)
        .filterDate(start_str, end_str)
    )

    def process_day(day_offset):
        current_date = start_date.advance(day_offset, 'day')
        next_date = current_date.advance(1, 'day')
        
        filtered = aod_coll.filterDate(current_date, next_date)
        has_images = filtered.size().gt(0)
        
        def calc_stats():
            daily_img = filtered.mean().multiply(0.001) # Scale factor for MAIAC AOD
            stats = daily_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=karachi_geometry,
                scale=1000,
                maxPixels=1e9
            )
            return ee.Feature(None, {
                "date": current_date.format("YYYY-MM-dd"),
                "aod_047_mean": stats.get("Optical_Depth_047")
            })
            
        def empty_stats():
            return ee.Feature(None, {"date": current_date.format("YYYY-MM-dd"), "aod_047_mean": None})

        return ee.Feature(ee.Algorithms.If(has_images, calc_stats(), empty_stats()))

    daily_fc = ee.FeatureCollection(day_list.map(process_day))
    features = daily_fc.getInfo()["features"]
    
    data_list = [
        f["properties"] for f in features 
        if f.get("properties") and f["properties"].get("aod_047_mean") is not None
    ]
    if data_list:
        df_year = pd.DataFrame(data_list)
        all_dfs.append(df_year)

# Concatenate all years
df_aod = pd.concat(all_dfs, ignore_index=True)
df_aod["date"] = pd.to_datetime(df_aod["date"])
df_aod = df_aod.sort_values("date").reset_index(drop=True)

# Save directly to Apache Parquet
output_parquet = os.path.join("data", "modis_aod_karachi_daily.parquet")
df_aod.to_parquet(output_parquet, engine="pyarrow", index=False)

print(f"\nSUCCESS: Exported {len(df_aod)} daily AOD records to: {output_parquet}")
print("\nFirst 5 rows of MODIS AOD Dataset:")
print(df_aod.head())