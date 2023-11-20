import numpy as np
import pandas as pd
from tensorflow.keras import layers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
#from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
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

   
def roberta_semantic_algorithm_news(csv_file):

    roberta = "fhamborg/roberta-targeted-sentiment-classification-newsarticles"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    scraped_data = pd.read_csv(csv_file)
    scraped_data = scraped_data.dropna()
    txt_articles = scraped_data['Text']

    semantic_articles = []
    count = 1

    for txt in txt_articles: 
        encoded_txt = tokenizer(txt, return_tensors='pt', max_length=512)
        output = model(**encoded_txt)

        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)

        semantic_scores = []
        for i in range(len(scores)):       
            s = scores[i]
            semantic_scores.append(s)        

        semantic_scores = np.array(semantic_scores)
        print(semantic_scores)
        semantic_articles.append(semantic_scores) 
        
        count += 1     
    scraped_data['Sentiment_RoBERTa'] = semantic_articles
    scraped_data.to_csv(csv_file, index=False)  # Saving the changes to the original CSV file
        

def lexicon_nltk(csv_file):
    #nltk.download('vader_lexicon')
    
    scraped_data = pd.read_csv(csv_file)
    scraped_data = scraped_data.dropna()
    txt_articles = scraped_data['Text']

    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = []

    for txt in txt_articles:
        sentiment_score = analyzer.polarity_scores(txt)['compound']
        sentiment_scores.append(sentiment_score)

    # Append sentiment scores as a new column to the existing DataFrame
    scraped_data['Sentiment_Lexicon'] = sentiment_scores

    # Save the updated DataFrame with sentiment scores to the CSV file
    scraped_data.to_csv(csv_file, index=False, encoding='utf-8')

def roberta_semantic_algorithm_twitter(csv_file):

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

    scraped_data = pd.read_csv(csv_file)
    #scraped_data = scraped_data[0:5]
    scraped_data = scraped_data.dropna()
    #scrapped_data = scrapped_data[scrapped_data['Text'].apply(lambda x: isinstance(x, str))]
    txt_articles = list(scraped_data['Text'])

    semantic_articles = []
    semantic_scores = []
    for txt in txt_articles: 

        encoded_txt = tokenizer(txt, return_tensors='pt', max_length=512)
        output = model(**encoded_txt)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        semantic_score = []
        for i in range(len(scores)):        
            l = labels[i]
            s = scores[i]
            semantic_score.append(s)        

        semantic_score = np.array(semantic_score)
        semantic = np.argmax(semantic_score)
        semantic_articles.append(semantic)
        semantic_scores.append(semantic_score)


            
        
    semantic_article_df = pd.DataFrame(semantic_articles)        
    semantic_article_df['Semantic'] = semantic_article_df[0].apply(semantic_label)
    semantic_articles_df = scraped_data.drop(columns=['Text'])
    semantic_articles_df['Semantic values roberta twitter'] = semantic_scores
    semantic_articles_df['Semantic roberta twitter'] = semantic_article_df['Semantic'].values

    file_name = os.path.basename(csv_file)
    parts = file_name.split('_')
    journal = parts[0]
    year = parts[1]
    base_path = os.path.dirname(csv_file)
    new_file_path = os.path.join(base_path, f'{journal}_{year}_semantics.csv')

    # Save the changes to the new CSV file
    semantic_articles_df.to_csv(new_file_path, index=False)

