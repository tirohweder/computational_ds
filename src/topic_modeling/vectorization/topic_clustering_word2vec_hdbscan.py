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

# Downloading NLTK's punkt tokenizer models
nltk.download('punkt')

# Tokenization function
def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    tokens = [token for token in tokens if token.isalpha() or token.isnumeric()]
    return tokens

# Setting up the directory to read CSV files
directory_output = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output')
df = pd.DataFrame()
path = os.path.join('*.csv')
frames = []
column_names = ["ID", "Headline", "Date", "Text", "Organization", "Link"]

# Reading and concatenating CSV files
for filename in glob.glob(path):
    df1 = pd.read_csv(filename, header=None, skiprows=1)
    frames.append(df1)
df = pd.concat(frames, ignore_index=True)
df.columns = column_names

# Loading the pre-trained Word2Vec model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Function to calculate average word vectors for a text
def average_word_vectors(text):
    tokens = tokenize(text)
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(300)

# Loading or generating Word2Vec embeddings
word2vec_embeddings_file = "word2vec_embeddings.npy"
if os.path.exists(word2vec_embeddings_file):
    vector_representations = np.load(word2vec_embeddings_file)
else:
    texts = df['Text'].tolist()
    vector_representations = np.array([average_word_vectors(text) for text in texts])
    np.save(word2vec_embeddings_file, vector_representations)

# Loading or generating UMAP embeddings
umap_embeddings_file = "umap_embeddings_word2vec.npy"
if os.path.exists(umap_embeddings_file):
    embedding = np.load(umap_embeddings_file)
else:
    reducer = umap.UMAP(n_components=3, random_state=42, low_memory=True)
    embedding = reducer.fit_transform(vector_representations)
    np.save(umap_embeddings_file, embedding)

# Loading or generating HDBSCAN cluster labels
hdbscan_labels_file = "hdbscan_cluster_labels_word2vec_15.npy"
if os.path.exists(hdbscan_labels_file):
    cluster_labels = np.load(hdbscan_labels_file)
else:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True, min_samples=15, cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(embedding)
    np.save(hdbscan_labels_file, cluster_labels)

    # Plotting Minimum Spanning Tree Visualization
    mst = clusterer.minimum_spanning_tree_.to_pandas()
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='viridis', s=50)
    for i, row in mst.iterrows():
        plt.plot([embedding[int(row['from']), 0], embedding[int(row['to']), 0]],
                 [embedding[int(row['from']), 1], embedding[int(row['to']), 1]], 'gray', alpha=0.3)
    plt.title('Minimum Spanning Tree Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig("minimum_spanning_tree_visualization_word2vec_15.png")
    plt.close()

# Preparing data for visualization and analysis
filtered_indices = np.where(cluster_labels != -1)[0]
df_filtered = df.iloc[filtered_indices].reset_index(drop=True)
filtered_embedding = embedding[filtered_indices]
filtered_cluster_labels = cluster_labels[filtered_indices]
filtered_texts = df_filtered['Text'].tolist()
filtered_headlines = df_filtered['Headline'].tolist()

# Create Plot Data
plot_data = pd.DataFrame(filtered_embedding, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
plot_data['Cluster'] = filtered_cluster_labels
plot_data['Text'] = filtered_texts
plot_data['Headline'] = filtered_headlines

# Calculating centroids and distances for clusters
centroids = [np.mean(filtered_embedding[filtered_cluster_labels == cluster], axis=0) for cluster in np.unique(filtered_cluster_labels)]
centroid_distances = euclidean_distances(centroids)
distance_matrix_df = pd.DataFrame(centroid_distances, index=np.unique(filtered_cluster_labels), columns=np.unique(filtered_cluster_labels))
distance_matrix_df.to_csv("centroid_distance_matrix_word2vec_15.csv")

# Assigning cluster labels to the original dataframe and saving
df['Cluster'] = cluster_labels
df['Cluster'] = df['Cluster'].fillna(-1)
df.to_csv('updated_dataframe_with_clusters_word2vec_15.csv', index=False)

# Plotting 3D scatter plot using Plotly
fig = px.scatter_3d(plot_data, x='UMAP_1', y='UMAP_2', z='UMAP_3', color='Cluster', hover_data=['Headline'], color_continuous_scale=px.colors.qualitative.Set1)
fig.update_layout(title='Word2Vec + HDBSCAN - minimum cluster size = 15 ', scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'))
fig.write_html("3d_plot_word2vec_15.html")
fig.show()
