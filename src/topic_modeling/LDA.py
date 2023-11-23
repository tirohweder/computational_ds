import gensim
import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import pyLDAvis
import pyLDAvis.gensim_models


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df=pd.read_csc("updated_dataframe_with_clusters_word2vec.csv")

def preprocess_text(df, text_column, custom_stopwords):
    clean_text_column = f"{text_column}_cleaned"
    lemmatizer = WordNetLemmatizer()

    def clean_single_text(text):
        if not isinstance(text, str):
            return []

        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in custom_stopwords]
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if not token.isdigit()]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token not in stop_words]

        return tokens

    # Apply the cleaning function to the specified column
    df[clean_text_column] = df[text_column].apply(clean_single_text)

    return df

# Define your custom stopwords and stop words
custom_stopwords = ["reuters", "fox", "cnn", "was", "has", "us", "year"]
stop_words = set(stopwords.words('english'))

# Call the function and update the DataFrame
df_clean_text = preprocess_text(df[:5], "Text", custom_stopwords)

# Save the DataFrame to a CSV file
df_clean_text.to_csv('cleaned_dataframe.csv', index=False, encoding='utf-8')

# Create a dictionary from the documents
dictionary = corpora.Dictionary(df_clean_text["Text_cleaned"])

# Create a corpus from the documents
corpus = [dictionary.doc2bow(doc) for doc in df_clean_text["Text_cleaned"]]

lda_model_F = LdaModel(corpus, num_topics=2000, id2word=dictionary, passes=10)

vis = pyLDAvis.gensim_models.prepare(lda_model_F, corpus, dictionary)

pyLDAvis.save_html(vis, 'lda_visualization.html')