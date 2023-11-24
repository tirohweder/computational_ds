import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
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

valid_vectors = []  # To store valid sentence vectorst
publication_colors = {}  # To store colors for each publication

for index, row in df.iterrows():
    sentence = str(row['Headline'])
    publication = str(row['Publication'])
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
            publication_colors[index] = publication  # Store the publication for coloring

# Convert the list of valid vectors into a NumPy array
valid_vectors = np.array(valid_vectors)

# UMAP dimensionality reduction
reducer = umap.UMAP(n_components=3, min_dist=0.1)  # increased min_dist from default
embedding_3d = reducer.fit_transform(valid_vectors)

# Create a new DataFrame with valid sentences and their labels
valid_df = df.iloc[:valid_vectors.shape[0]].copy()

# Create a list of colors based on the publications
colors = [publication_colors.get(index, None) for index in valid_df.index]

# Create 3D scatter plot
fig = px.scatter_3d(
    valid_df, x=embedding_3d[:, 0], y=embedding_3d[:, 1], z=embedding_3d[:, 2],
    color=colors, hover_data=['Headline'],
    color_discrete_map={pub: px.colors.qualitative.Set1[i] for i, pub in enumerate(df['Publication'].unique())}
)

# Display the plot
fig.show()
