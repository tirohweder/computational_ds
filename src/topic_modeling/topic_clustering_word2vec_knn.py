import gensim
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from nltk.tokenize import word_tokenize
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

import hdbscan

# We decide no to do stemming or removing stop words as the large pretrained model by google has emeddings for all types of words
nltk.download('punkt')

# Load the Google News Word2Vec model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Function to average word vectors for a text
def average_word_vectors(text):
    tokens = word_tokenize(text.lower())
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)  # Return a zero vector if no words are in the model

# Load your dataset
df = pd.read_csv('/home/timm/Projects/computational_ds/data/output/cnn_2014_output.csv')
df = df.head(10000)
texts = df['Text'].tolist()

# Get vector representation for each text
vector_representations = np.array([average_word_vectors(text) for text in texts])

# UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(vector_representations)

#K-Means clustering on the reduced data
num_clusters = 100
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedding)


# Calculate distances from each point to its cluster center
distances = np.min(kmeans.transform(embedding), axis=1)

# Determine the threshold for filtering (e.g., 95th percentile)
threshold = np.percentile(distances, 95)

filtered_indices = np.where(distances < threshold)[0]

# Reset the index of the original DataFrame to align with the filtered data
df_filtered = df.iloc[filtered_indices].reset_index(drop=True)

filtered_embedding = embedding[filtered_indices]
filtered_cluster_labels = cluster_labels[filtered_indices]
filtered_texts = df_filtered['Text'].tolist()
filtered_headlines = df_filtered['Headline'].tolist()  # Assuming 'Headline' column exists

davies_bouldin_idx = davies_bouldin_score(filtered_embedding, filtered_cluster_labels)

# Prepare data for Plotly visualization
plot_data = pd.DataFrame(filtered_embedding, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
plot_data['Cluster'] = filtered_cluster_labels
plot_data['Text'] = filtered_texts
plot_data['Headline'] = filtered_headlines  # Add headlines to the plot data

if len(np.unique(filtered_cluster_labels)) > 1:
    silhouette = silhouette_score(filtered_embedding, filtered_cluster_labels)
    print("Silhouette Score:", silhouette)
else:
    print("Silhouette Score cannot be computed with a single cluster.")
# 3D Plotting using Plotly
fig = px.scatter_3d(
    plot_data,
    x='UMAP_1', y='UMAP_2', z='UMAP_3',
    color='Cluster',
    hover_data=['Headline'],  # Use headlines for hover text
    color_continuous_scale=px.colors.qualitative.Set1
)

fig.update_layout(
    title='KNN + Word2Vec 0.7602479672845226, sil: 0.44417053',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    )
)

fig.show()

print(davies_bouldin_idx)