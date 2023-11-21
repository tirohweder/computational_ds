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

df_final=pd.read_csv("/cleaned_dataframe.csv")
df_final["Text_cleaned"] = df_final["Text_cleaned"].apply(lambda x: eval(x))

# Create a dictionary from the documents
dictionary = corpora.Dictionary(df_final["Text_cleaned"])

# Create a corpus from the documents
corpus = [dictionary.doc2bow(doc) for doc in df_final["Text_cleaned"]]

lda_model_F = LdaModel(corpus, num_topics=2000, id2word=dictionary, passes=5)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model_F, corpus, dictionary)
pyLDAvis.display(vis)