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
df = pd.read_csv('/home/timm/Projects/computational_ds/data/output/cnn_2014_output.csv')  # Update the path to your dataset
texts = df['Text'].tolist()

# Text vectorization with custom tokenizer
vectorizer = TfidfVectorizer(max_features=5000, tokenizer=tokenize_and_lemmatize)
X = vectorizer.fit_transform(texts)

# UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(X.toarray())

# HDBSCAN clustering on the reduced data
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(embedding)

filtered_indices = np.where(cluster_labels != -1)[0]

# Reset the index of the original DataFrame to align with the filtered data
df_filtered = df.iloc[filtered_indices].reset_index(drop=True)

filtered_embedding = embedding[filtered_indices]
filtered_cluster_labels = cluster_labels[filtered_indices]
filtered_texts = df_filtered['Text'].tolist()
filtered_headlines = df_filtered['Headline'].tolist()  # Assuming 'Headline' column exists

# Prepare data for Plotly visualization
plot_data = pd.DataFrame(filtered_embedding, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
plot_data['Cluster'] = filtered_cluster_labels
plot_data['Text'] = filtered_texts
plot_data['Headline'] = filtered_headlines  # Add headlines to the plot data

# 3D Plotting using Plotly
fig = px.scatter_3d(
    plot_data,
    x='UMAP_1', y='UMAP_2', z='UMAP_3',
    color='Cluster',
    hover_data=['Headline'],  # Use headlines for hover text
    color_continuous_scale=px.colors.qualitative.Set1
)


fig.update_layout(
    title='HDBSCAN + TFIDF with Noise Removal',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    )
)

fig.show()
