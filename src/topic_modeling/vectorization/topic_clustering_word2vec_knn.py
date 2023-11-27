import gensim
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from nltk.tokenize import word_tokenize
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import hdbscan

# Download NLTK's punkt tokenizer models for word tokenization
nltk.download('punkt')

# Load the pre-trained Google News Word2Vec model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Function to average word vectors for a given text
def average_word_vectors(text):
    tokens = word_tokenize(text.lower())  # Tokenizing text
    vectors = [model[word] for word in tokens if word in model]  # Getting vectors for tokens
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(300)  # Averaging vectors

# Load dataset
df = pd.read_csv('/home/timm/Projects/computational_ds/data/output/cnn_2014_output.csv')
df = df.head(10000)  # Limiting to first 10000 rows for processing
texts = df['Text'].tolist()

# Generate vector representations for each text
vector_representations = np.array([average_word_vectors(text) for text in texts])

# Dimensionality reduction using UMAP
reducer = umap.UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(vector_representations)

# K-Means clustering on the reduced data
num_clusters = 100
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedding)

# Calculate distances from each point to its cluster center
distances = np.min(kmeans.transform(embedding), axis=1)

# Determine the threshold for filtering outliers
threshold = np.percentile(distances, 95)
filtered_indices = np.where(distances < threshold)[0]

# Filter the dataset and embeddings based on the calculated threshold
df_filtered = df.iloc[filtered_indices].reset_index(drop=True)
filtered_embedding = embedding[filtered_indices]
filtered_cluster_labels = cluster_labels[filtered_indices]
filtered_texts = df_filtered['Text'].tolist()
filtered_headlines = df_filtered['Headline'].tolist()  # Assuming 'Headline' column exists

# Calculate clustering evaluation metrics
davies_bouldin_idx = davies_bouldin_score(filtered_embedding, filtered_cluster_labels)

# Prepare data for visualization
plot_data = pd.DataFrame(filtered_embedding, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
plot_data['Cluster'] = filtered_cluster_labels
plot_data['Text'] = filtered_texts
plot_data['Headline'] = filtered_headlines

# Compute silhouette score if more than one cluster
if len(np.unique(filtered_cluster_labels)) > 1:
    silhouette = silhouette_score(filtered_embedding, filtered_cluster_labels)
    print("Silhouette Score:", silhouette)
else:
    print("Silhouette Score cannot be computed with a single cluster.")

# 3D Plotting using Plotly
fig = px.scatter_3d(
    plot_data, x='UMAP_1', y='UMAP_2', z='UMAP_3', color='Cluster',
    hover_data=['Headline'], color_continuous_scale=px.colors.qualitative.Set1
)
fig.update_layout(
    title='KNN + Word2Vec 0.7602479672845226, sil: 0.44417053',
    scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3')
)
fig.show()

# Print Davies-Bouldin Index
print(davies_bouldin_idx)
