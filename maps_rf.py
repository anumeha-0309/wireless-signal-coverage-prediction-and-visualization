import joblib
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

rf_model = joblib.load("rsrp_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("preprocessed_main_dataset_3.csv")

center_lat = df["Latitude"].mean()
center_lon = df["Longitude"].mean()

buffer = 0.01
min_lat, max_lat = df["Latitude"].min() - buffer, df["Latitude"].max() + buffer
min_lon, max_lon = df["Longitude"].min() - buffer, df["Longitude"].max() + buffer

grid_res = 0.0005

lats = np.arange(min_lat, max_lat, grid_res)
lons = np.arange(min_lon, max_lon, grid_res)

grid_points = [(lat, lon) for lat in lats for lon in lons]
print(f"Generating predictions for {len(grid_points)} grid points...")

grid_df = pd.DataFrame(grid_points, columns=["Latitude", "Longitude"])

avg_elev = df["Elevation"].mean()
grid_df["Elevation"] = avg_elev

grid_df["lat_lon_interact"] = grid_df["Latitude"] * grid_df["Longitude"]

features = ["Latitude", "Longitude", "Elevation", "lat_lon_interact"]

grid_scaled = scaler.transform(grid_df[features])

grid_rsrp = rf_model.predict(grid_scaled)
grid_df["RSRP"] = grid_rsrp

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

heat_data = [[row["Latitude"], row["Longitude"], -row["RSRP"]] for _, row in grid_df.iterrows()]
HeatMap(heat_data, radius=8, blur=12).add_to(m)

m.save("localized_rsrp_heatmap_rf.html")
print("Heatmap saved as 'localized_rsrp_heatmap_rf.html'")

print(f"Average predicted RSRP: {grid_df['RSRP'].mean():.2f} dBm")
print(f"Good signal (> -100 dBm): {sum(grid_df['RSRP'] >= -100)}/{len(grid_df)}")

grid_df.to_csv("localized_rsrp_predictions_rf.csv", index=False)
print("Predictions exported to 'localized_rsrp_predictions_rf.csv'")
