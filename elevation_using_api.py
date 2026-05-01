import pandas as pd
import requests
import numpy as np
from time import sleep

def get_elevation_batch(latitudes, longitudes, max_retries=3, backoff_factor=2):
    url = "https://api.open-elevation.com/api/v1/lookup"
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in zip(latitudes, longitudes)]
    data = {"locations": locations}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            api_data = response.json()
            results = api_data.get("results", [])
            elevations = [result.get("elevation") for result in results]
            if len(elevations) == len(latitudes) and all(e is not None for e in elevations):
                return elevations
            else:
                raise ValueError("Incomplete or invalid response from API")
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                sleep(sleep_time)
            else:
                print("Max retries exceeded. Returning None values.")
                return [None] * len(latitudes)

def update_elevations_from_api(df, batch_size=100, output_file="updated_main_dataset_3.csv"):
    required_cols = ["Latitude", "Longitude", "Elevation"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain Latitude, Longitude, and Elevation columns.")
    
    valid_coords_mask = ~(df["Latitude"].isna() | df["Longitude"].isna())
    invalid_coords_count = (~valid_coords_mask).sum()
    if invalid_coords_count > 0:
        print(f"Warning: {invalid_coords_count} rows have invalid coordinates (NaN lat/lon). These will be skipped.")
    
    valid_coords = df[valid_coords_mask][["Latitude", "Longitude"]].reset_index(drop=True)
    total_to_fill = len(valid_coords)
    
    if total_to_fill == 0:
        print("No valid coordinates found.")
        df.to_csv(output_file, index=False)
        return df
    
    print(f"Updating elevations for all {total_to_fill} valid rows using API.")
    
    elevations = []
    for i in range(0, total_to_fill, batch_size):
        batch = valid_coords.iloc[i:i + batch_size]
        batch_lats = batch["Latitude"].tolist()
        batch_lons = batch["Longitude"].tolist()
        batch_elevations = get_elevation_batch(batch_lats, batch_lons)
        elevations.extend(batch_elevations)
        print(f"Processed batch {i // batch_size + 1}: {len(batch)} coordinates")
        sleep(1)  
    
    valid_indices = df[valid_coords_mask].index
    df.loc[valid_indices, "Elevation"] = elevations
    
    still_missing_count = df["Elevation"].isna().sum()
    filled_count = len(df) - still_missing_count
    print(f"Successfully filled {filled_count} elevations. {still_missing_count} still empty (API failures or invalid coords).")
    
    df.to_csv(output_file, index=False)
    print(f"Updated DataFrame saved to {output_file}")
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("main_dataset_3.csv")
    df_updated = update_elevations_from_api(df, output_file="updated_main_dataset_3.csv")
    
    print("\nUpdated DataFrame:")
    print(df_updated.head())