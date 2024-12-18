import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point

# Load datasets
crime_data = pd.read_csv("sampled_crime_data.csv")
urban_data = pd.read_csv("filtered_urban_data.csv")

# =====================
# Preprocess Crime Data
# =====================
crime_data['CMPLNT_FR_DT'] = pd.to_datetime(crime_data['CMPLNT_FR_DT'])
crime_data['Year'] = crime_data['CMPLNT_FR_DT'].dt.year
crime_data = crime_data.dropna(subset=['Latitude', 'Longitude'])
crime_features = crime_data[['Latitude', 'Longitude', 'Year']]

# =====================
# Preprocess Urban Data
# =====================
urban_data['Filing Date'] = pd.to_datetime(urban_data['Filing Date'])
urban_data['Year'] = urban_data['Filing Date'].dt.year
urban_data = urban_data.dropna(subset=['LATITUDE', 'LONGITUDE'])
urban_features = urban_data[['LATITUDE', 'LONGITUDE', 'Year']]

# Standardize the features for clustering
scaler = StandardScaler()
crime_scaled = scaler.fit_transform(crime_features)
urban_scaled = scaler.fit_transform(urban_features)

# ===========================
# Apply K-Means Clustering
# ===========================
k = 5  # Number of clusters

# Crime Clustering
kmeans_crime = KMeans(n_clusters=k, random_state=42)
crime_data['Cluster'] = kmeans_crime.fit_predict(crime_scaled)

# Urban Project Clustering
kmeans_urban = KMeans(n_clusters=k, random_state=42)
urban_data['Cluster'] = kmeans_urban.fit_predict(urban_scaled)

# ============================
# Calculate Redundancy Metrics
# ============================

def calculate_redundancy(data, cluster_column, lat_col, lon_col):
    """
    Calculate redundancy for each cluster based on shared spatial overlap.
    Redundancy is computed as the average density of points within each cluster
    relative to its neighbors.
    """
    redundancy_scores = {}
    for cluster in data[cluster_column].unique():
        # Points in the current cluster
        cluster_points = data[data[cluster_column] == cluster]
        cluster_area = len(cluster_points)  # Number of points in this cluster

        # Points in other clusters
        other_points = data[data[cluster_column] != cluster]

        # Merge based on latitude and longitude to calculate overlaps
        overlap = cluster_points.merge(
            other_points,
            how='inner',
            left_on=[lat_col, lon_col],
            right_on=[lat_col, lon_col]
        ).shape[0]

        # Redundancy as percentage overlap
        redundancy_scores[cluster] = overlap / cluster_area if cluster_area > 0 else 0

    return redundancy_scores

# Redundancy for Crime Clusters
crime_redundancy = calculate_redundancy(
    crime_data, cluster_column='Cluster', lat_col='Latitude', lon_col='Longitude'
)
print("Crime Cluster Redundancy:", crime_redundancy)

# Redundancy for Urban Development Clusters
urban_redundancy = calculate_redundancy(
    urban_data, cluster_column='Cluster', lat_col='LATITUDE', lon_col='LONGITUDE'
)
print("Urban Development Cluster Redundancy:", urban_redundancy)

# ================================
# Convert to GeoDataFrames for Map
# ================================
crime_gdf = gpd.GeoDataFrame(
    crime_data,
    geometry=[Point(xy) for xy in zip(crime_data['Longitude'], crime_data['Latitude'])],
    crs="EPSG:4326"
)
urban_gdf = gpd.GeoDataFrame(
    urban_data,
    geometry=[Point(xy) for xy in zip(urban_data['LONGITUDE'], urban_data['LATITUDE'])],
    crs="EPSG:4326"
)

# ============================
# Side-by-Side Visualization
# ============================
fig, axes = plt.subplots(2, 1, figsize=(10, 18), sharex=True, sharey=True)

# Plot Urban Development Clusters
ax1 = axes[0]
urban_colors = ['blue', 'cyan', 'navy', 'purple', 'darkblue']
for i in range(k):
    urban_gdf[urban_gdf['Cluster'] == i].plot(
        ax=ax1, color=urban_colors[i], markersize=5, label=f'Cluster {i}\nRedundancy: {urban_redundancy[i]:.2f}'
    )
ax1.set_title('Urban Development Clusters')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(markerscale=3)

# Plot Crime Clusters
ax2 = axes[1]
crime_colors = ['red', 'orange', 'brown', 'darkred', 'salmon']
for i in range(k):
    crime_gdf[crime_gdf['Cluster'] == i].plot(
        ax=ax2, color=crime_colors[i], markersize=1, label=f'Cluster {i}\nRedundancy: {crime_redundancy[i]:.2f}'
    )
ax2.set_title('Crime Clusters')
ax2.set_xlabel('Longitude')
ax2.legend(markerscale=3)

# Adjust layout
plt.tight_layout()
plt.show()