import networkx as nx
import pandas as pd
from itertools import combinations

# Load datasets
crime_data = pd.read_csv('sampled_crime_data.csv')
urban_data = pd.read_csv('filtered_urban_data.csv')

# Group the data by BORO_NM and OFNS_DESC to compute counts
crime_counts = crime_data.groupby(['BORO_NM', 'OFNS_DESC']).size().reset_index(name='count')

# Create a bipartite graph
G = nx.Graph()
for _, row in crime_counts.iterrows():
    G.add_edge(row['BORO_NM'], row['OFNS_DESC'], weight=row['count'])

# Project to a unipartite crime graph
crime_nodes = crime_counts['OFNS_DESC'].unique()
crime_graph = nx.bipartite.weighted_projected_graph(G, crime_nodes)

# Compute clustering coefficients on the projected graph
clustering_coeffs = nx.clustering(crime_graph, weight='weight')

# Compute redundancy on the projected graph
redundancy = {}
for node in crime_graph.nodes():
    neighbors = list(crime_graph.neighbors(node))
    if len(neighbors) < 2:
        redundancy[node] = 0
        continue
    redundancy[node] = sum(
        len(set(crime_graph.neighbors(neighbor)) & set(neighbors)) for neighbor in neighbors
    ) / len(neighbors)

# Sort and format clustering coefficients
sorted_clustering = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)
top_clustering = {k: round(v, 2) for k, v in sorted_clustering[:10]}

# Sort and format redundancy values
sorted_redundancy = sorted(redundancy.items(), key=lambda x: x[1], reverse=True)
top_redundancy = {k: round(v, 2) for k, v in sorted_redundancy[:10]}

# Display formatted and restricted results
print("Top 10 Clustering Coefficients (Crime-Type Graph):")
print(top_clustering)

print("\nTop 10 Redundancy Values (Crime-Type Graph):")
print(top_redundancy)