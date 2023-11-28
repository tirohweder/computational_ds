import pandas as pd
import networkx as nx
import os
import plotly.graph_objects as go
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors


def count_articles(cluster_pair):
    count = combined_df[(combined_df['Organization'] == cluster_pair[1][4:]) & (combined_df['Cluster'].apply(lambda x: str(cluster_pair[0]) in str(x)))].shape[0]
    return count


def edge_width(cluster_pair, max_count):

    count = count_articles(cluster_pair)
    # Define the range for widths (0.5 to 8)
    min_width = 0.5
    max_width = 8.0

    # Linear interpolation to scale the width between min and max based on count
    scaled_width = min_width + (max_width - min_width) * (count / max_count)

    return scaled_width

def edge_color(cluster_pair):    

    filtered_df = combined_df[
        (combined_df['Organization'] == cluster_pair[1][4:]) & (combined_df['Cluster'].apply(lambda x: str(cluster_pair[0]) in str(x)))
    ]

    if not filtered_df.empty:

        # Access the 'Semantic values roberta' column
        semantic_values = filtered_df['Sentiment value lexicon'].tolist()

        semantic = sum(semantic_values)/len(semantic_values)
        
        # Define a colormap ranging from red (negative) to white (neutral) to green (positive)
        cmap = mcolors.LinearSegmentedColormap.from_list('sentiment_gradient', ['#ff0000', '#ffffff', '#00ff00'])
  
        # Map values to colors in the defined colormap
        colors = mcolors.to_hex(cmap(semantic))
        
        return colors
    else:
        return 'grey'  # Default color if sentiment information is missing or edge not found

     
base_path = r""

# Load the distance matrix
distance_df = pd.read_csv(os.path.join("centroid_distance_matrix_word2vec_15.csv"), index_col=0)

# Create a network graph
G = nx.Graph()

# Load your data
combined_df = pd.read_csv(os.path.join(
    "../topic_modeling/vectorization/updated_dataframe_with_clusters_and_semantics.csv"))

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

degrees = dict(G.degree())
max_count = max(degrees.values())

print(max_count)

# Use MDS to compute the positions
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=6)
mds_pos = mds.fit_transform(distance_df)
pos = {str(cluster): pos for cluster, pos in zip(distance_df.columns, mds_pos)}
pos.update(org_positions)

# Extract edge information
edge_x = []
edge_y = []
#edge_widths = []
edge_traces = []



for edge in G.edges():
    if edge[0] in pos and edge[1] in pos:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        #edge_x.extend([x0, x1, None])
        #edge_y.extend([y0, y1, None])
        width = edge_width(edge, max_count)
        color = edge_color(edge)
        #edge_widths.extend([width, width, None])
        edge_trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None], line=dict(width=width, color=color), hoverinfo='none', mode='lines')
        edge_traces.append(edge_trace)

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
fig = go.Figure(data=[*edge_traces, node_trace],
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
