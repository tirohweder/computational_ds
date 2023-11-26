import numpy as np
import pandas as pd
from tensorflow.keras import layers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

#labeling function for sentiment values
def semantic_label(value):
    if value < -0.33:
        label = 'negative'
    elif value < 0.33:
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
        
#vader lexicon library for sentiment analysis that outputs one value [-1, 1] classifying as negative, neutral or positive
def lexicon_nltk(csv_file):
    
    #nltk.download('vader_lexicon')
    
    scraped_data = pd.read_csv(csv_file)
    txt_articles = scraped_data['Text']

    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = []

    for txt in txt_articles:
        sentiment_score = analyzer.polarity_scores(txt)
        sentiment_score = np.array(list(sentiment_score.values())[0:3])
        sentiment_scores.append(sentiment_score)


    semantic_articles_df = scraped_data.drop(columns=['Text'])
    semantic_articles_df['Sentiment scores lexicon'] = sentiment_scores

    # Calculate weighted sentiment values
    semantic_articles_df['Sentiment value lexicon'] = check_weighted_sum_consistency(semantic_articles_df)

    # Apply semantic_label to categorize sentiment values
    semantic_articles_df['Sentiment lexicon'] = semantic_articles_df['Sentiment value lexicon'].apply(semantic_label)

    file_name = os.path.basename(csv_file)
    parts = file_name.split('_')
    journal = parts[0]
    year = parts[1]
    base_path = os.path.dirname(csv_file)
    new_file_path = os.path.join(base_path, f'{journal}_{year}_semantics_lex.csv')

    # Save the changes to the new CSV file
    semantic_articles_df.to_csv(new_file_path, index=False)


#roberta model for sentiment analysis that outputs three values [0, 1] weighting each negative, neutral or positive label
def roberta_semantic_algorithm_twitter(csv_file):

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

    scraped_data = pd.read_csv(csv_file)
    scraped_data = scraped_data[scraped_data['Organization']=='Reuters']

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
    #parts = file_name.split('_')
    #journal = parts[0]
    #year = parts[1]
    base_path = os.path.dirname(csv_file)
    new_file_path = os.path.join(base_path, f'reuters_final_semantics_rob.csv')
    #new_file_path = os.path.join(base_path, f'{journal}_{year}_semantics_rob.csv')

    # Save the changes to the new CSV file
    semantic_articles_df.to_csv(new_file_path, index=False)


#vector transformation into a single value between -1 and 1 of semantic values given by roberta
#done for comparison between the lexicon model
def check_weighted_sum_consistency(df):

    df['Semantic values roberta twitter'] = df['Semantic values roberta twitter'].apply(lambda x: np.array([float(value) for value in x.strip('[]').split()])) 
    coefficients = {
        'Negative': 1,
        'Neutral': 2,
        'Positive': 3
    }

    # Initialize a list to store calculated sentiments based on weighted sum
    sentiment_value = []

    # Calculate sentiment based on weighted sum for each row
    for values in df['Semantic values roberta twitter']: #df['Sentiment scores lexicon']:
        # Calculate the weighted sum using values and weights
        weighted_sum = np.sum(values * np.array([coefficients['Negative'], coefficients['Neutral'], coefficients['Positive']]))

        # Perform vector transformation
        vector_trans_value = weighted_sum - 2

        sentiment_value.append(vector_trans_value)

    sentiment_series = pd.Series(sentiment_value, name='Sentiment values roberta twitter')#'Sentiment value lexicon'
    return sentiment_series

def sampling_articles(csv_file):
    df = pd.read_csv(csv_file)
    sample_csv = df.sample(n=5000)
    return sample_csv

def merge_data(large_master_file_df, new_data_df, column_1 = 'Sentiment value lexicon', column_2 = 'Sentiment lexicon', column_3= 'Semantic roberta twitter', column_4='Sentiment values roberta'):

    new_data_df = new_data_df.loc[:,['Headline', column_1, column_2, column_3, column_4]]

    master_file_merged = pd.merge(large_master_file_df, new_data_df, on='Headline', how='left')

    return master_file_merged
