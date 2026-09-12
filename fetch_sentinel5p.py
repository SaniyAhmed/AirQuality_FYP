import os
import geopandas as gpd
import ee

# 1. Initialize Google Earth Engine
PROJECT_ID = "green-cover-karachi"
ee.Initialize(project=PROJECT_ID)

# 2. Load local Karachi boundary
geojson_path = os.path.join("data", "karachi_only.geojson")
gdf = gpd.read_file(geojson_path)

min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
karachi_geometry = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

# 3. Define Multi-Year Date Range (3-Year Historical Window)
start_date = "2023-01-01"
end_date = "2026-09-01"  # Fetching data right up to recent months

print("--- FETCHING MULTI-YEAR SENTINEL-5P DATA ---")
print(f"Date Range: {start_date} to {end_date}")

# Query Sentinel-5P NO2 collection
no2_collection = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
    .select("NO2_column_number_density")
    .filterBounds(karachi_geometry)
    .filterDate(start_date, end_date)
)

total_passes = no2_collection.size().getInfo()
print(f"Total Sentinel-5P daily passes found over Karachi: {total_passes}")

# Calculate multi-year average concentration
mean_no2 = no2_collection.mean().clip(karachi_geometry)
stats = mean_no2.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=karachi_geometry,
    scale=1000,
    maxPixels=1e9
).getInfo()

print("\nMulti-Year Mean Tropospheric NO2 (mol/m²):")
print(stats)