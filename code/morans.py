# Moran's I spatial autocorrelation was attempted but failed due to python version issues

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pysal.explore.esda import Moran, Moran_Local
from pysal.lib import weights
from shapely.geometry import Point, Polygon
from pysal.viz.splot.esda import moran_scatterplot, lisa_cluster

# Load crime and urban development datasets
crime_data = pd.read_csv("crime_data.csv")
urban_data = pd.read_csv("urban_data.csv")

# Drop rows with missing latitude/longitude
crime_data = crime_data.dropna(subset=["Latitude", "Longitude"])
urban_data = urban_data.dropna(subset=["LATITUDE", "LONGITUDE"])

# Convert to GeoDataFrames
crime_gdf = gpd.GeoDataFrame(crime_data,
                             geometry=gpd.points_from_xy(crime_data['Longitude'], crime_data['Latitude']),
                             crs="EPSG:4326")

urban_gdf = gpd.GeoDataFrame(urban_data,
                             geometry=gpd.points_from_xy(urban_data['LONGITUDE'], urban_data['LATITUDE']),
                             crs="EPSG:4326")

# Create a grid for spatial aggregation
grid_size = 0.01  # Grid resolution (degrees)
xmin, ymin, xmax, ymax = crime_gdf.total_bounds
x_coords = np.arange(xmin, xmax, grid_size)
y_coords = np.arange(ymin, ymax, grid_size)

grid_cells = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)])
              for x in x_coords for y in y_coords]
grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")

# Spatial join to count crime and urban projects within grid cells
grid['crime_count'] = gpd.sjoin(grid, crime_gdf, how="left", predicate="intersects").groupby(grid.index).size()
grid['urban_count'] = gpd.sjoin(grid, urban_gdf, how="left", predicate="intersects").groupby(grid.index).size()

# Fill missing values with 0
grid['crime_count'] = grid['crime_count'].fillna(0)
grid['urban_count'] = grid['urban_count'].fillna(0)

# Spatial Weights Matrix for Moran's I
w = weights.KNN.from_dataframe(grid, k=8)
w.transform = 'R'

# Moran's I - Global Spatial Autocorrelation for Crime Counts
moran_global = Moran(grid['crime_count'], w)
print(f"Global Moran's I: {moran_global.I:.4f}, p-value: {moran_global.p_sim:.4f}")

# Local Moran's I for identifying spatial clusters
moran_local = Moran_Local(grid['crime_count'], w)

# Visualization of Local Moran's I (LISA Clusters)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# LISA cluster map
grid['lisa_clusters'] = moran_local.q
grid.plot(column='lisa_clusters', cmap='coolwarm', legend=True, ax=ax[0])
ax[0].set_title("Local Moran's I Clusters (Crime Density)")

# Overlay urban development density
grid.plot(column='urban_count', cmap='Blues', legend=True, ax=ax[1])
ax[1].set_title("Urban Development Density Heatmap")

plt.tight_layout()
plt.show()

# Moran's I Scatter Plot
fig, ax = plt.subplots(figsize=(7, 5))
moran_scatterplot(moran_global, ax=ax)
ax.set_title("Global Moran's I Scatter Plot for Crime Density")
plt.show()