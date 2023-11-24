import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

# Load the distance matrix
distance_df = pd.read_csv('centroid_distance_matrix_word2vec.csv', index_col=0)

# Create a network graph
G = nx.Graph()

# Load your data
combined_df = pd.read_csv('updated_dataframe_with_clusters_word2vec.csv')
combined_df = combined_df[combined_df['Cluster'] != -1]

# Count articles per cluster and normalize cluster sizes
cluster_counts = combined_df['Cluster'].value_counts()
scaler = MinMaxScaler(feature_range=(5, 15))  # Adjust range according to your preference
normalized_sizes = scaler.fit_transform(cluster_counts.values.reshape(-1, 1)).flatten()

# Map cluster labels to their normalized sizes
cluster_sizes = {str(cluster): size for cluster, size in zip(cluster_counts.index, normalized_sizes)}

# Add cluster nodes based on the distance matrix
for cluster in distance_df.columns:
    G.add_node(str(cluster), type='cluster', label=str(cluster), size=cluster_sizes.get(str(cluster), 5))

# Add organization nodes and define positions
org_positions = {'Org_CNN': (-15, 0), 'Org_FOX': (0, -10), 'Org_Reuters': (15, 0)}
for org in combined_df['Organization'].unique():
    G.add_node(f"Org_{org}", type='org', label=org)

    # Get the subset of the dataframe for the current organization
    org_df = combined_df[combined_df['Organization'] == org]

    # Iterate over the unique clusters for the current organization
    for cluster in org_df['Cluster'].unique():
        G.add_edge(f"Org_{org}", str(cluster))

# Use MDS to compute the positions
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=6)
mds_pos = mds.fit_transform(distance_df)
pos = {str(cluster): pos for cluster, pos in zip(distance_df.columns, mds_pos)}
pos.update(org_positions)

# Extract edge information
edge_x = []
edge_y = []
for edge in G.edges():
    if edge[0] in pos and edge[1] in pos:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

# Extract node information and adjust sizes
node_x = []
node_y = []
node_text = []
node_color = []
node_size = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

    if G.nodes[node]['type'] == 'org':
        node_text.append(G.nodes[node]['label'])
        node_color.append('#1273de')
        node_size.append(30)  # Fixed size for organization nodes
    else:
        node_text.append('')  # No label for cluster nodes
        node_color.append('#FF5733')
        node_size.append(G.nodes[node].get('size', 5))

node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='bottom center',
                        hoverinfo='text', marker=dict(showscale=False, color=node_color, size=node_size, line_width=2))

# Create a figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Organization-Cluster Sentiment Network',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

fig.show()
