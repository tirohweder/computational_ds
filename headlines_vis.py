import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import hdbscan
import umap.umap_ as umap
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import plotly.express as px

import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load Word2Vec model (you can load your own or use the provided GoogleNews model)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Read the DataFrame containing headlines
df = pd.read_csv('headlines.csv')
df = df.head(10000)
# Tokenize and preprocess text for each sentence
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

valid_vectors = []  # To store valid sentence vectors

for index, row in df.iterrows():
    sentence = str(row['Headline'])
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Check if there are valid tokens left after preprocessing
    if tokens:
        # Create a sentence vector by averaging word vectors
        sentence_vector = np.mean([model[word] for word in tokens if word in model], axis=0)
        if sentence_vector.shape == (300,):  # Check the shape
            valid_vectors.append(sentence_vector)
# Convert the list of valid vectors into a NumPy array
valid_vectors = np.array(valid_vectors)

for title in valid_vectors:
    print(title.shape)
# UMAP dimensionality reduction
reducer = umap.UMAP(n_components=3, min_dist=0)  # increased min_dist from default
embedding_3d = reducer.fit_transform(valid_vectors)

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
labels = clusterer.fit_predict(embedding_3d)

# Create a new DataFrame with valid sentences and their labels
valid_df = df.iloc[:valid_vectors.shape[0]].copy()
valid_df['label'] = labels

# Create 3D scatter plot
fig = px.scatter_3d(
    valid_df, x=embedding_3d[:, 0], y=embedding_3d[:, 1], z=embedding_3d[:, 2],
    color='label', text='Publication', hover_data=['Headline'],
    color_continuous_scale=px.colors.qualitative.Set1
)

# Display the plot
fig.show()