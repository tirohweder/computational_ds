# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:44:01 2023

@author: storr
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
from scipy.linalg import svd



data = pd.read_csv(r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\updated_dataframe_with_clusters_word2vec.csv')

data = data[0:100]

lemmatizer = WordNetLemmatizer()

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []
    wordnet_tagged = list(wordnet_tagged)
    
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            
    return " ".join(lemmatized_sentence)



def tf_idf_matrix_creater(data):
    
    text_list = []
    
    
    for article in range(len(data)):
        text = data.loc[article,'Text']
        try:
            if not pd.isna(text):
                text = text.lower()
                text = lemmatize_sentence(text)
                text_list.append(text)
        except:
            pass
    
        
    tf_idf = TfidfVectorizer(max_df=0.8)
    
    result = tf_idf.fit_transform(text_list)
    tf_idf_matrix = result.toarray()
    
        
    return tf_idf_matrix
    
    


def svd_transform(tf_idf_matrix):
    
    svd = TruncatedSVD(n_components = 100)
    
  
    return svd.fit_transform(tf_idf_matrix)
    






#%%    

tf_idf_matrix = tf_idf_matrix_creater(data)
     
svd_tf_idf_matrix = svd_transform(tf_idf_matrix)

u,s,vt = svd(tf_idf_matrix)

docs_similarities = cosine_similarity(svd_tf_idf_matrix,svd_tf_idf_matrix)

#retreive 10 closest articles
closest_articles= np.argsort(docs_similarities, axis = 1)[:,-2:-12:-1]