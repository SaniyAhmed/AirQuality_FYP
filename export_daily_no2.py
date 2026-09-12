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

print("--- EXPORTING DAILY SENTINEL-5P DATA TO PARQUET ---")

for year in years:
    start_str = f"{year}-01-01"
    end_str = f"{year}-12-31" if year < 2026 else "2026-09-01"
    
    print(f"Processing year: {year}...")
    
    start_date = ee.Date(start_str)
    end_date = ee.Date(end_str)
    n_days = end_date.difference(start_date, 'day')
    day_list = ee.List.sequence(0, n_days.subtract(1))
    
    no2_coll = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        .select("NO2_column_number_density")
        .filterBounds(karachi_geometry)
        .filterDate(start_str, end_str)
    )

    def process_day(day_offset):
        current_date = start_date.advance(day_offset, 'day')
        next_date = current_date.advance(1, 'day')
        
        filtered = no2_coll.filterDate(current_date, next_date)
        
        # Check if satellite images exist for this day
        has_images = filtered.size().gt(0)
        
        # Define reduction if images exist
        def calc_stats():
            daily_img = filtered.mean()
            stats = daily_img.reduceRegion(
                reducer=ee.Reducer.mean()
                .combine(reducer2=ee.Reducer.min(), sharedInputs=True)
                .combine(reducer2=ee.Reducer.max(), sharedInputs=True),
                geometry=karachi_geometry,
                scale=1000,
                maxPixels=1e9
            )
            return ee.Feature(None, {
                "date": current_date.format("YYYY-MM-dd"),
                "no2_mean": stats.get("NO2_column_number_density_mean"),
                "no2_min": stats.get("NO2_column_number_density_min"),
                "no2_max": stats.get("NO2_column_number_density_max")
            })
            
        # Return empty feature if no satellite pass / heavy cloud cover
        def empty_stats():
            return ee.Feature(None, {"date": current_date.format("YYYY-MM-dd"), "no2_mean": None})

        return ee.Feature(ee.Algorithms.If(has_images, calc_stats(), empty_stats()))

    # Map daily sequence and retrieve FeatureCollection
    daily_fc = ee.FeatureCollection(day_list.map(process_day))
    features = daily_fc.getInfo()["features"]
    
    # Filter out empty/null days
    data_list = [
        f["properties"] for f in features 
        if f.get("properties") and f["properties"].get("no2_mean") is not None
    ]
    if data_list:
        df_year = pd.DataFrame(data_list)
        all_dfs.append(df_year)

# Concatenate all years into a single DataFrame
df_final = pd.concat(all_dfs, ignore_index=True)
df_final["date"] = pd.to_datetime(df_final["date"])
df_final = df_final.sort_values("date").reset_index(drop=True)

# Save directly to Apache Parquet
output_parquet = os.path.join("data", "sentinel5p_karachi_daily.parquet")
df_final.to_parquet(output_parquet, engine="pyarrow", index=False)

print(f"\nSUCCESS: Exported {len(df_final)} clean daily records to: {output_parquet}")
print("\nFirst 5 rows of Parquet dataset:")
print(df_final.head())