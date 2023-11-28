import os
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import plotly.express as px
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

''' This script was used for testing feasibility of clustering using TFIDF and KNN. '''


# Downloading NLTK data for tokenization and lemmatization
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

# Loading dataset from a CSV file
directory_output = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output')
df = pd.read_csv(directory_output, 'cnn_2014_output.csv')
texts = df['Text'].tolist()

# Vectorizing text using TFIDF with custom tokenizer
vectorizer = TfidfVectorizer(max_features=5000, tokenizer=tokenize_and_lemmatize)
X = vectorizer.fit_transform(texts)

# Dimensionality reduction using UMAP
reducer = umap.UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(X.toarray())

# Clustering with K-Means
num_clusters = 100
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedding)

# Calculating distances to cluster centers and filtering outliers
distances = np.min(kmeans.transform(embedding), axis=1)
threshold = np.percentile(distances, 95)
filtered_indices = np.where(distances < threshold)[0]
df_filtered = df.iloc[filtered_indices].reset_index(drop=True)

# Preparing filtered data for visualization and clustering metrics
filtered_embedding = embedding[filtered_indices]
filtered_cluster_labels = cluster_labels[filtered_indices]
filtered_texts = df_filtered['Text'].tolist()
filtered_headlines = df_filtered['Headline'].tolist()  # Assuming 'Headline' column exists
davies_bouldin_idx = davies_bouldin_score(filtered_embedding, filtered_cluster_labels)

# Creating a 3D scatter plot using Plotly
plot_data = pd.DataFrame(filtered_embedding, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
plot_data['Cluster'] = filtered_cluster_labels
plot_data['Text'] = filtered_texts
plot_data['Headline'] = filtered_headlines
fig = px.scatter_3d(
    plot_data, x='UMAP_1', y='UMAP_2', z='UMAP_3', color='Cluster',
    hover_data=['Headline'], color_continuous_scale=px.colors.qualitative.Set1
)
fig.update_layout(
    title='KNN + TFIDF 15 - Davies= 0.5204151351174, Silhouette: 0.5733855',
    scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3')
)
fig.show()

# Printing Davies-Bouldin Index
print(davies_bouldin_idx)

# Calculating Silhouette Score
if len(np.unique(filtered_cluster_labels)) > 1:
    silhouette = silhouette_score(filtered_embedding, filtered_cluster_labels)
    print("Silhouette Score:", silhouette)
else:
    print("Silhouette Score cannot be computed with a single cluster.")
