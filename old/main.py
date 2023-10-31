

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec, KeyedVectors

import umap.umap_ as umap
import pandas as pd
import hdbscan
import plotly.express as px

import matplotlib.pyplot as plt
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Train Word2Vec model
#model = Word2Vec(processed_sentences, vector_size=100, window=5, min_count=1, workers=4)
#model.save("word2vec.model")

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vector = model['computer']


#with open('ats_corpus/heroesheroinesof00ritc.txt', 'r') as file:
#    raw_text = file.read()

df = pd.read_csv('headlines.csv')
raw_text = df['Headline']

# Tokenize and preprocess text
sentences = sent_tokenize(raw_text)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

processed_sentences = []

for sentence in sentences:
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_sentences.append(tokens)



# Fetching vector for a sample word
#vector = model.wv['sample']
#print(vector)
words = []
vectors = []

for sentence in processed_sentences:
    for word in sentence:
        if word in model:
            words.append(word)
            vectors.append(model[word])

####


reducer = umap.UMAP(n_components=3, min_dist=0) # increased min_dist from default
embedding_3d = reducer.fit_transform(vectors)

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
labels = clusterer.fit_predict(embedding_3d)

# Convert embeddings to DataFrame for plotting with Plotly
df = pd.DataFrame(embedding_3d, columns=['x', 'y', 'z'])
df['word'] = words
df['label'] = labels  # add cluster labels to the dataframe

# Create 3D scatter plot
fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', text='word', hover_data=['word'], color_continuous_scale=px.colors.qualitative.Set1)

# Display the plot
fig.show()
