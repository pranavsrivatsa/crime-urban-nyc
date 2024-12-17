import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load datasets
crime_data = pd.read_csv('sampled_crime_data.csv')
urban_data = pd.read_csv('filtered_urban_data.csv')

# Group by OFNS_DESC and BORO_NM, count occurrences
crime_cooccur = crime_data.groupby(['OFNS_DESC', 'BORO_NM']).size().reset_index(name='count')

# Identify the top 10 crimes by total count across boroughs
top_crimes = (
    crime_cooccur.groupby('OFNS_DESC')['count']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

# Filter the data to include only the top crimes
filtered_crime_cooccur = crime_cooccur[crime_cooccur['OFNS_DESC'].isin(top_crimes)]

# Create a graph
G = nx.Graph()

# Add nodes and weighted edges
for _, row in filtered_crime_cooccur.iterrows():
    crime = row['OFNS_DESC']
    borough = row['BORO_NM']
    weight = row['count']
    G.add_edge(crime, borough, weight=weight)

# Customize node properties
node_colors = []
node_sizes = []
for node in G.nodes:
    if node in top_crimes:
        node_colors.append('skyblue')  # Crime types
        node_sizes.append(4000)  # Larger size for crime types
    else:
        node_colors.append('lightgreen')  # Boroughs
        node_sizes.append(3000)  # Larger size for boroughs

# Customize edge properties
edges = G.edges(data=True)
edge_weights = [edge[2]['weight'] for edge in edges]  # Edge weights for thickness

# Draw the graph
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42)  # Improved layout for better spacing
nx.draw(
    G, pos, with_labels=True, node_size=node_sizes, font_size=12, font_color='black',
    node_color=node_colors, edge_color='gray', width=[0.1 + w / max(edge_weights) * 2 for w in edge_weights],
    font_weight='semibold'  # Set font weight to bold
)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_weight='bold')

# Add legend
legend_elements = [
    Patch(facecolor='skyblue', edgecolor='black', label='Crime Type (Top 10)'),
    Patch(facecolor='lightgreen', edgecolor='black', label='Borough')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=12)

# Title
plt.title("Top 10 Crimes and Their Borough Associations", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
