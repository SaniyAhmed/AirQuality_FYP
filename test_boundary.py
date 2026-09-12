import os
import geopandas as gpd

# Relative paths
input_path = os.path.join("data", "karachi_boundary.geojson")
output_path = os.path.join("data", "karachi_only.geojson")

if not os.path.exists(input_path):
    print(f"Error: Could not find {input_path}")
else:
    # 1. Load the full Pakistan boundary file
    full_gdf = gpd.read_file(input_path)

    # 2. Filter rows where 'adm2_name' contains 'Karachi'
    karachi_gdf = full_gdf[full_gdf["adm2_name"].str.contains("Karachi", case=False, na=False)]

    print("--- KARACHI FILTERING RESULTS ---")
    print("Number of Karachi districts found:", len(karachi_gdf))
    print("Districts included:")
    print(karachi_gdf["adm2_name"].tolist())

    # 3. Save the filtered GeoJSON for future pipeline steps
    karachi_gdf.to_file(output_path, driver="GeoJSON")
    print(f"\nSuccessfully saved clean boundary file to: {output_path}")

    # 4. Extract the EXACT Karachi Bounding Box
    karachi_bounds = karachi_gdf.total_bounds
    print("\nTRUE Karachi Bounding Box [Min Lon, Min Lat, Max Lon, Max Lat]:")
    print(karachi_bounds)