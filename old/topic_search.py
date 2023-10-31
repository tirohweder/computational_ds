import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec, KeyedVectors
import umap.umap_ as umap
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load Word2Vec model (you can load your own or use the provided GoogleNews model)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Read the DataFrame containing headlines
df = pd.read_csv('headlines.csv')
#df = df.head(10000)

# Collect user input
user_input = input("Enter a sentence: ")

# Tokenize and preprocess text for each headline
stop_words = set(stopwords.words('english'))

headline_vectors = []  # To store headline vectors
publication_colors = {}  # To store colors for each publication

for index, row in df.iterrows():
    sentence = str(row['Headline'])
    publication = str(row['Publication'])
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]

    # Create a list of word vectors for valid tokens
    word_vectors = [model[word] for word in tokens if word in model]

    if word_vectors:
        # Create a sentence vector by averaging word vectors
        sentence_vector = np.mean(word_vectors, axis=0)
        headline_vectors.append(sentence_vector)
        publication_colors[index] = publication  # Store the publication for coloring

if not headline_vectors:
    print("No valid headlines found for the given user input.")
else:
    # Convert the list of headline vectors into a NumPy array
    headline_vectors = np.array(headline_vectors)

    # Calculate cosine similarity between user input and headlines
    user_input_vector = np.mean([model[word] for word in user_input.split() if word in model], axis=0)
    similarities = cosine_similarity([user_input_vector], headline_vectors)

    # Get indices of top 10 most similar headlines
    top_indices = np.argsort(similarities[0])[::-1][:10]

    # Create a DataFrame with the top 10 similar headlines and their labels
    valid_df = df.iloc[top_indices].copy()

    # Create a list of colors based on the publications for the selected headlines
    selected_colors = [publication_colors[index] for index in valid_df.index]

    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=3, min_dist=0)
    embedding_3d = reducer.fit_transform(headline_vectors[top_indices])  # Only use the vectors of selected headlines

    # Create 3D scatter plot
    fig = px.scatter_3d(
        valid_df, x=embedding_3d[:, 0], y=embedding_3d[:, 1], z=embedding_3d[:, 2],
        color=selected_colors, hover_data=['Headline', 'Publication']
    )

    # Display the plot
    fig.show()