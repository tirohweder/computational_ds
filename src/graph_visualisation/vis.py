import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.manifold import MDS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load the distance matrix
distance_df = pd.read_csv('centroid_distance_matrix_word2vec.csv', index_col=0)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
distance_imputed = imputer.fit_transform(distance_df)
distance_df_imputed = pd.DataFrame(distance_imputed, index=distance_df.index, columns=distance_df.columns)

# Create a network graph
G = nx.Graph()

# Load your combined CSV data
combined_df = pd.read_csv('updated_dataframe_with_clusters_word2vec.csv')

# Count articles per cluster
cluster_counts = combined_df['Cluster'].value_counts()

# Normalize cluster sizes
scaler = MinMaxScaler(feature_range=(5, 15))
normalized_sizes = scaler.fit_transform(cluster_counts.values.reshape(-1, 1)).flatten()

# Map cluster labels to their normalized sizes
cluster_sizes = {str(cluster): size for cluster, size in zip(cluster_counts.index, normalized_sizes)}

# Add cluster nodes
for cluster in distance_df_imputed.columns:
    G.add_node(str(cluster), type='cluster', size=cluster_sizes.get(str(cluster), 5))

# Extract unique organizations
unique_orgs = combined_df['Organization'].unique()

# Add organization nodes and define initial positions
org_positions = {}
for idx, org in enumerate(unique_orgs):
    org_positions[f"Org_{org}"] = (idx, 0)
    G.add_node(f"Org_{org}", type='org', label=org)

# Add edges between organization nodes and clusters
for _, row in combined_df.iterrows():
    org_node = f"Org_{row['Organization']}"
    cluster_node = str(row['Cluster'])
    if cluster_node in distance_df_imputed.columns:
        G.add_edge(org_node, cluster_node)

# Use MDS to compute the positions
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=6)
mds_pos = mds.fit_transform(distance_df_imputed)
pos = {str(cluster): pos for cluster, pos in zip(distance_df_imputed.columns, mds_pos)}
pos.update(org_positions)

# Edge and node information extraction
edge_x, edge_y, node_x, node_y, node_text, node_color, node_size = [], [], [], [], [], [], []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

    if G.nodes[node]['type'] == 'org':
        node_text.append(G.nodes[node]['label'])
        node_color.append('#1273de')
        node_size.append(10)  # Fixed size for organization nodes
    else:
        node_text.append('')  # No label for cluster nodes
        node_color.append('#FF5733')
        node_size.append(G.nodes[node].get('size', 5))

node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='bottom center',
                        hoverinfo='text', marker=dict(showscale=False, color=node_color, size=node_size, line_width=2))

# Create and display the figure
fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
    title='<br>Organization-Cluster Sentiment Network',
    titlefont_size=16, showlegend=False, hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))


fig.write_html('network_graph.html')
fig.show()

