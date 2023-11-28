import pandas as pd
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import plotly.express as px
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import save_npz, load_npz
import glob
import os

# Setting location for file loading
directory_output = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output')

# Downloading NLTK data required for text processing
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initializing NLTK lemmatizer and stopwords for English language
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to tokenize, lemmatize, and clean text data
def tokenize_and_lemmatize(text):
    # Tokenizing and converting to lowercase
    tokens = word_tokenize(text.lower())
    # Removing non-alphanumeric characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    # Lemmatizing and filtering out stopwords and non-alphabetic tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

# Reading and concatenating data files from the specified directory
df = pd.DataFrame()
path = os.path.join(directory_output, '*.csv')
frames = []
column_names = ["ID", "Headline", "Date", "Text", "Organization", "Link"]
for filename in glob.glob(path):
    df1 = pd.read_csv(filename, header=None, skiprows=1)
    frames.append(df1)
df = pd.concat(frames, ignore_index=True)
df.columns = column_names

# Generating or loading TFIDF embeddings
tfidf_embeddings_file = "tfidf_embeddings.npz"
if os.path.exists(tfidf_embeddings_file):
    X = load_npz(tfidf_embeddings_file)
else:
    vectorizer = TfidfVectorizer(max_features=5000, tokenizer=tokenize_and_lemmatize, max_df=0.8)
    texts = df['Text'].tolist()
    X = vectorizer.fit_transform(texts)
    save_npz(tfidf_embeddings_file, X)

# Generating or loading UMAP embeddings
umap_embeddings_file = "umap_embeddings_tfidf.npy"
if os.path.exists(umap_embeddings_file):
    embedding = np.load(umap_embeddings_file)
else:
    reducer = umap.UMAP(n_components=3, random_state=42, low_memory=True)
    embedding = reducer.fit_transform(X.toarray())
    np.save(umap_embeddings_file, embedding)

# Performing or loading HDBSCAN clustering
hdbscan_labels_file = "hdbscan_cluster_labels_tfidf_15.npy"
if os.path.exists(hdbscan_labels_file):
    cluster_labels = np.load(hdbscan_labels_file)
else:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)
    np.save(hdbscan_labels_file, cluster_labels)

# Filtering and preparing data for visualization
filtered_indices = np.where(cluster_labels != -1)[0]
df_filtered = df.iloc[filtered_indices].reset_index(drop=True)
filtered_embedding = embedding[filtered_indices]
filtered_cluster_labels = cluster_labels[filtered_indices]
filtered_texts = df_filtered['Text'].tolist()
filtered_headlines = df_filtered['Headline'].tolist()

# Preparing plot data with UMAP embeddings and cluster labels
plot_data = pd.DataFrame(filtered_embedding, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
plot_data['Cluster'] = filtered_cluster_labels
plot_data['Text'] = filtered_texts
plot_data['Headline'] = filtered_headlines

# Calculating centroids of clusters and their pairwise distances
centroids = []
unique_clusters = np.unique(filtered_cluster_labels)
for cluster in unique_clusters:
    cluster_points = filtered_embedding[filtered_cluster_labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)
centroid_distances = euclidean_distances(centroids)
distance_matrix_df = pd.DataFrame(centroid_distances, index=unique_clusters, columns=unique_clusters)
distance_matrix_df.to_csv("centroid_distance_matrix_tfidf_15.csv")

# Assigning cluster labels to the original dataframe
df['Cluster'] = cluster_labels
df['Cluster'] = df['Cluster'].fillna(-1)
df.to_csv('updated_dataframe_with_clusters_tfidf_15.csv', index=False)

# Plotting 3D scatter plot using Plotly
fig = px.scatter_3d(plot_data, x='UMAP_1', y='UMAP_2', z='UMAP_3', color='Cluster', hover_data=['Headline'], color_continuous_scale=px.colors.qualitative.Set1)
fig.update_layout(
    title='HDBSCAN + TFIDF 15',
    scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3')
)
fig.write_html("3d_plot_tfidf_15.html")
fig.show()