import numpy as np
import pandas as pd
from tensorflow.keras import layers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
#from textblob import TextBlob
import nltk
#import bert
#!pip install bert-for-tf2
#!pip install sentencepiece
#try:
#    %tensorflow_version 2.x
#except Exception:
#    pass
#import tensorflow as tf

#import tensorflow_hub as hub


def semantic_label(value):
    if value == 0:
        label = 'negative'
    elif value == 1:
        label = 'neutral'    
    else:
        label = 'positive'

    return label  

   

def roberta_semantic_algorithm(scrapped_data):

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

    scrapped_data = scrapped_data.dropna()
    #scrapped_data = scrapped_data[scrapped_data['Text'].apply(lambda x: isinstance(x, str))]
    txt_articles = list(scrapped_data['Text'])

    semantic_articles = []

    for txt in txt_articles: 

        encoded_txt = tokenizer(txt, return_tensors='pt', max_length=512)
        output = model(**encoded_txt)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        semantic_scores = []
        for i in range(len(scores)):        
            l = labels[i]
            s = scores[i]
            semantic_scores.append(s)        

        semantic_scores = np.array(semantic_scores)
        semantic = np.argmax(semantic_scores)
        semantic_articles.append(semantic)   
            
        
    semantic_article_df = pd.DataFrame(semantic_articles)        
    semantic_article_df['Semantic'] = semantic_article_df[0].apply(semantic_label)
    semantic_articles_df = scrapped_data.drop(columns=['Text'])
    semantic_articles_df['Semantic'] = semantic_article_df['Semantic'].values

    return semantic_articles_df

def lexicon(scraped_data):

    scraped_data = scraped_data.dropna()
    txt_articles = list(scraped_data['Text'])

    semantic_articles = []

    for txt in txt_articles: 

        blob = TextBlob(txt)

        # Perform sentiment analysis
        sentiment_score = blob.sentiment.polarity

        # Interpret the sentiment score
        if sentiment_score > 0:
            semantic_articles.append("Positive")
        elif sentiment_score < 0:
            semantic_articles.append("Negative")
        else:
            semantic_articles.append("Neutral")
        
    return    pd.DataFrame(semantic_articles)        

