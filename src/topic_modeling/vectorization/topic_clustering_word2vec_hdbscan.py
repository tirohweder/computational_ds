import pandas as pd
import numpy as np
import nltk
import umap
import plotly.express as px
import re
import os
import glob
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
# Download necessary NLTK data
nltk.download('punkt')

# Custom tokenizer function
def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    tokens = [token for token in tokens if token.isalpha() or token.isnumeric()]
    return tokens

# Load your dataset
directory_output = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output')
df = pd.DataFrame()
path = os.path.join('*.csv')
frames = []

column_names = ["ID", "Headline", "Date", "Text", "Organization", "Link", "Sentiment"]

for filename in glob.glob(path):
    df1 = pd.read_csv(filename, header=None, skiprows=1)
    frames.append(df1)

df = pd.concat(frames, ignore_index=True)
df.columns = column_names

print(df)
texts = df['Text'].tolist()

# Load Word2Vec Model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Function to create a vector for a document
def average_word_vectors(text):
    tokens = tokenize(text)
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)  # Return a zero vector if no words are in the model

# Check if the Word2Vec embeddings file exists
word2vec_embeddings_file = "old/word2vec_embeddings.npy"
if os.path.exists(word2vec_embeddings_file):
    print("Found Word2Vec Embeddings")
    vector_representations = np.load(word2vec_embeddings_file)
else:
    vector_representations = np.array([average_word_vectors(text) for text in texts])
    np.save(word2vec_embeddings_file, vector_representations)

# Check if the UMAP embeddings file exists
umap_embeddings_file = "umap_embeddings_word2vec.npy"
if os.path.exists(umap_embeddings_file):
    print("Found Umap Embeddings")
    embedding = np.load(umap_embeddings_file)
else:
    print("Reducing with Umap")
    reducer = umap.UMAP(n_components=3, random_state=42, low_memory=True)
    embedding = reducer.fit_transform(vector_representations)
    np.save(umap_embeddings_file, embedding)

# HDBSCAN clustering on the reduced data
hdbscan_labels_file = "hdbscan_cluster_labels_word2vec.npy"
if os.path.exists(hdbscan_labels_file):
    cluster_labels = np.load(hdbscan_labels_file)
    print("Found Cluster Labels")
else:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True, min_samples=15,
                                cluster_selection_method='eom', cluster_selection_epsilon=0)
    cluster_labels = clusterer.fit_predict(embedding)
    np.save(hdbscan_labels_file, cluster_labels)

    mst = clusterer.minimum_spanning_tree_.to_pandas()

    # Plotting the MST
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='viridis', s=50)
    for i, row in mst.iterrows():
        from_idx = int(row['from'])
        to_idx = int(row['to'])
        plt.plot([embedding[from_idx, 0], embedding[to_idx, 0]],
                 [embedding[from_idx, 1], embedding[to_idx, 1]], 'gray', alpha=0.3)

    plt.title('Minimum Spanning Tree Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot to a file
    plt.savefig("minimum_spanning_tree_visualization.png")

    plt.close()

filtered_indices = np.where(cluster_labels != -1)[0]

# Reset the index of the original DataFrame to align with the filtered data
df_filtered = df.iloc[filtered_indices].reset_index(drop=True)

filtered_embedding = embedding[filtered_indices]
filtered_cluster_labels = cluster_labels[filtered_indices]
filtered_texts = df_filtered['Text'].tolist()
filtered_headlines = df_filtered['Headline'].tolist()

# Prepare data for Plotly visualization
plot_data = pd.DataFrame(filtered_embedding, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
plot_data['Cluster'] = filtered_cluster_labels
plot_data['Text'] = filtered_texts
plot_data['Headline'] = filtered_headlines

centroids = []
unique_clusters = np.unique(filtered_cluster_labels)
for cluster in unique_clusters:
    cluster_points = filtered_embedding[filtered_cluster_labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

# Calculate pairwise distances between centroids
centroid_distances = euclidean_distances(centroids)

# Convert distances to a DataFrame for better readability
distance_matrix_df = pd.DataFrame(centroid_distances,
                                  index=unique_clusters,
                                  columns=unique_clusters)

# Save the DataFrame to a CSV file
distance_matrix_df.to_csv("centroid_distance_matrix_word2vec.csv")

df['Cluster'] = cluster_labels

# For the rows where clustering was not performed or the data was filtered out,
# you can assign a default value such as -1 or 'Not Clustered'
df['Cluster'] = df['Cluster'].fillna(-1)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_dataframe_with_clusters_word2vec_15.csv', index=False)

# 3D Plotting using Plotly
fig = px.scatter_3d(
    plot_data,
    x='UMAP_1', y='UMAP_2', z='UMAP_3',
    color='Cluster',
    hover_data=['Headline'],
    color_continuous_scale=px.colors.qualitative.Set1
)

# Davies-Bouldin: 0.32418409691808064, Silhouette = 0.7323601 for cnn_2014
fig.update_layout(
    title='Word2Vec + HDBSCAN - minimum cluster size = 15 ',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    )
)
fig.write_html("3d_plot_word2vec_15.html")

fig.show()
