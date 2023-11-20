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
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Lemmatizer and Stop Words Initialization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom tokenizer function
def tokenize_and_lemmatize(text):
    # Tokenization and lowercasing
    tokens = word_tokenize(text.lower())
    # Removing punctuation and special characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    # Lemmatizing and filtering stop words and non-alphabetic tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

# Load your dataset
df = pd.read_csv('/home/timm/Projects/computational_ds/data/output/cnn_2014_output.csv')
#df = df.head(10000)
texts = df['Text'].tolist()

# Text vectorization with custom tokenizer
vectorizer = TfidfVectorizer(max_features=5000, tokenizer=tokenize_and_lemmatize)
X = vectorizer.fit_transform(texts)

# UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(X.toarray())

# K-Means clustering on the reduced data
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
    title='KNN + TFIDF with Outlier Removal 0.5204151351174, sil: 0.5733855',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    )
)

fig.show()

print(davies_bouldin_idx)