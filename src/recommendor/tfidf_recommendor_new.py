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

#%%

data = pd.read_csv(r'updated_dataframe_with_clusters_word2vec.csv')


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
        #print(text)
        try:
            if not pd.isna(text):
                text = text.lower()
                text = lemmatize_sentence(text)
                text_list.append(text)
        except:
            pass
    
        
    tf_idf = TfidfVectorizer(max_df = 0.9)
    
    result = tf_idf.fit_transform(text_list)
    tf_idf_matrix = result.toarray()
    
        
    return tf_idf_matrix
    
    


def svd_transform(tf_idf_matrix):
    
    svd = TruncatedSVD(n_components = 100)
    
  
    return svd.fit_transform(tf_idf_matrix)
    






#%%    

clusters = max(data['Cluster'].values)
cluster_count = data['Cluster'].value_counts()

#Retrieve clusters with more than 100 articles 
cluster_more_than_100_articles = cluster_count[cluster_count>=100].index

#Select 10 largest clusters
top_10_biggest_clusters = list(cluster_more_than_100_articles[1:10+1])
 


#%%
closest_articles_dict = {}
for cluster in top_10_biggest_clusters:
    
    #closest_articles_list = []
    data_cluster = data[data['Cluster'] == cluster].reset_index(drop=True)
    tf_idf_matrix = tf_idf_matrix_creater(data_cluster)
         
    svd_tf_idf_matrix = svd_transform(tf_idf_matrix)    
    
    docs_similarities = cosine_similarity(svd_tf_idf_matrix,svd_tf_idf_matrix)
    
    #retreive 10 closest articles
    closest_articles = np.argsort(docs_similarities, axis = 1)[:,-2:-12:-1]
    
    #closest_articles_list.append(closest_articles)

    closest_articles_dict[cluster] = closest_articles
    
    
#%%   

#Converting recommended articles into a df to save results
closest_articles_df = pd.DataFrame()

for cluster in top_10_biggest_clusters:
    
    closest_articles_per_cluster = closest_articles_dict[cluster]
    closest_articles_per_cluster_df = pd.DataFrame(closest_articles_per_cluster)
    closest_articles_per_cluster_df['Cluster'] = cluster
    closest_articles_df = closest_articles_df.append(closest_articles_per_cluster_df).reset_index(drop=True)

closest_articles_df.to_csv(r'recommender_for_biggest_clusters.csv', index=False)


#%%

#Merging data from master file with the recommendation to a new df (recommended_articles_clusters_df)
recommended_articles_clusters_df = pd.DataFrame()

for cluster in top_10_biggest_clusters:
    
    data_cluster = data[data['Cluster'] == cluster].reset_index(drop=True)
    recommended_cluster = closest_articles_df[closest_articles_df['Cluster']==cluster].reset_index(drop=True)
    recommendor_cluster = pd.merge(data_cluster, recommended_cluster, left_index=True, right_index=True, how='left')
    recommended_articles_clusters_df = recommended_articles_clusters_df.append(recommendor_cluster)
    
#%%

#Out of the 10 recommended articles for each article, select those that are from different news organizations
recommended_articles_clusters_df['recommended article from opposing org 1'] = ''
recommended_articles_clusters_df['recommended article from opposing org 2'] = ''


for cluster in top_10_biggest_clusters:
    
    recommended_articles_cluster = recommended_articles_clusters_df[recommended_articles_clusters_df['Cluster_x']==cluster]
    for i in range(len(recommended_articles_cluster)):
        
        if recommended_articles_cluster['Organization'].iloc[i] or recommended_articles_cluster['Link'].iloc[i] == 'CNN':
            org1 = 'FOX'
            org2 = 'Reuters'
        elif recommended_articles_cluster['Organization'].iloc[i] or recommended_articles_cluster['Link'].iloc[i] == 'FOX':
            org1 = 'CNN'
            org2 = 'Reuters'
        else:
            org1='CNN'
            org2='FOX'
            
        org1_extracted = 0
        org2_extracted = 0
        
        
        for j in range(10):
            
            #if articles from both organizations are stored already, skip to next row (i.e next article)
            if org1_extracted==1 and org2_extracted==1:
                continue
            
            article = recommended_articles_cluster.loc[i,j]
            
            #some articles where found to have the information in the 'Organization' and 'Link' column interchanged thus the 'or' statement
            if org1 in recommended_articles_cluster['Organization'].iloc[article]  or org1 in recommended_articles_cluster['Link'].iloc[article]:
                recommended_articles_clusters_df.at[i, 'recommended article from opposing org 1'] = article
                org1_extracted = 1
                
            if org2 in recommended_articles_cluster['Organization'].iloc[article]  or org2 in recommended_articles_cluster['Link'].iloc[article]:
                recommended_articles_clusters_df.at[i, 'recommended article from opposing org 2'] = article
                org2_extracted = 1    
            
        recommended_articles_clusters_df['recommended article from opposing org 1'] = recommended_articles_clusters_df['recommended article from opposing org 1'].append(recommended_articles_cluster['recommended article from opposing org 1'],ignore_index=True).reset_index(drop=True)
        recommended_articles_clusters_df['recommended article from opposing org 2'] = recommended_articles_clusters_df['recommended article from opposing org 2'].append(recommended_articles_cluster['recommended article from opposing org 2'],ignore_index=True).reset_index(drop=True)
                
