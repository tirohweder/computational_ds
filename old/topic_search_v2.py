import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import umap.umap_ as umap

# Load the pre-trained model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def get_average_word2vec(tokens, model, vector_size):
    """
    Compute the average word vector for a list of tokens.
    """
    vec = np.zeros(vector_size)
    count = 0.0
    for word in tokens:
        try:
            vec += model[word]
            count += 1.0
        except KeyError:  # if the word is not in the model's vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

df = pd.read_csv('headlines.csv')
df = df.head(100000)

headlines = df['Headline'].tolist()
# Tokenize headlines
tokenized_headlines = [str(headline).split() for headline in headlines]
# Convert headlines to average word vectors
headline_embeddings = [get_average_word2vec(tokens, model, 300) for tokens in tokenized_headlines]

# Convert keyword to its word vector
keyword = "climate"
keyword_embedding = model[keyword].reshape(1, -1)

# Compute similarities
similarities = cosine_similarity(keyword_embedding, headline_embeddings)



# Rank headlines by similarity
sorted_indices = similarities.argsort()[0][::-1]

best_matching_headlines = [headlines[i] for i in sorted_indices]

reducer_3d = umap.UMAP(n_components=3)
reduced_embeddings_3d = reducer_3d.fit_transform(headline_embeddings)
reduced_keyword_embedding_3d = reducer_3d.transform(keyword_embedding)

# Calculate cosine similarity between the 3D reduced keyword_embedding and the reduced_embeddings_3d
similarities_3d = cosine_similarity(reduced_keyword_embedding_3d, reduced_embeddings_3d)

# Add this new similarity to the df_embedding_3d DataFrame


# Create a DataFrame for the reduced embeddings
df_embedding_3d = pd.DataFrame(reduced_embeddings_3d, columns=['x', 'y', 'z'])

# Add the similarity values and hover text
df_embedding_3d['similarity_3d'] = similarities_3d[0]
df_embedding_3d['similarity'] = similarities[0]
df_embedding_3d['hover_text'] = df_embedding_3d.apply(
    lambda row: f"Headline: {headlines[row.name]}\nSimilarity before reduction: {row['similarity']:.4f}\nSimilarity after 3D reduction: {row['similarity_3d']:.4f}",
    axis=1
)
df_filtered = df_embedding_3d[df_embedding_3d['similarity'] > 0.2]

# 2. Create a 3D scatter plot
fig = px.scatter_3d(df_filtered, x='x', y='y', z='z', color='similarity',
                    color_continuous_scale="Viridis",
                    title="3D UMAP projection of headline embeddings",
                    labels={"similarity": "Cosine Similarity"},
                    hover_name='hover_text')

fig.show()


print(best_matching_headlines[:10])