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
        
#vader lexicon library for sentiment analysis that outputs one value [-1, 1] classifying as negative, neutral or positive
def lexicon_nltk(csv_file):
    
    nltk.download('vader_lexicon')
    
    scraped_data = pd.read_csv(csv_file)
    scraped_data = scraped_data[0:5]
    scraped_data = scraped_data.dropna()
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
    semantic_articles_df['Sentiment value lexicon'] = check_weighted_sum_consistency(csv_file)
    semantic_articles_df['Sentiment lexicon'] = semantic_articles_df[0].apply(semantic_label)

    file_name = os.path.basename(csv_file)
    parts = file_name.split('_')
    journal = parts[0]
    year = parts[1]
    base_path = os.path.dirname(csv_file)
    new_file_path = os.path.join(base_path, f'{journal}_{year}_semantics.csv')

    # Save the changes to the new CSV file
    semantic_articles_df.to_csv(new_file_path, index=False)




#roberta model for sentiment analysis that outputs three values [0, 1] weighting each negative, neutral or positive label
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


#vector transformation into a single value between -1 and 1 of semantic values given by roberta
#done for comparison between the lexicon model
def check_weighted_sum_consistency(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract values from column 6
    sentiment_values = df.iloc[:, 6].apply(lambda x: [float(val) for val in x[1:-1].split()])
    print(sentiment_values)
    # Extract actual sentiments from column 7
    actual_sentiments = df.iloc[:, 7]  
    
    # Assign weights to sentiment categories
    coeffients = {
        'Negative': 1,
        'Neutral': 2,
        'Positive': 3
    }
    
    # Initialize a list to store calculated sentiments based on weighted sum
    sentiment_value = []
    
    # Calculate sentiment based on weighted sum for each row
    for values in sentiment_values:
        # Calculate the weighted sum using values and weights
        weighted_sum = sum(value * coeffients[sentiment] for value, sentiment in zip(values, ['Negative', 'Neutral', 'Positive']))
        print(weighted_sum)       
        # Determine the sentiment category based on the weighted sum
        '''if weighted_sum < 1:
            calculated_sentiments.append('negative')
        elif weighted_sum < 2:
            calculated_sentiments.append('neutral')
        else:
            calculated_sentiments.append('positive')'''
        vector_trans_value = weighted_sum -2
        print(vector_trans_value)
        sentiment_value.append(vector_trans_value)
    # Check consistency between calculated sentiments and actual sentiments
    #consistencies = [calculated == actual for calculated, actual in zip(calculated_sentiments, actual_sentiments)]
    
    #true_count = consistencies.count(True)
    # #false_count = consistencies.count(False)

    return sentiment_value
