import numpy as np
import pandas as pd
from tensorflow.keras import layers

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
#!pip install bert-for-tf2
#!pip install sentencepiece
#try:
#    %tensorflow_version 2.x
#except Exception:
#    pass
#import tensorflow as tf

#import tensorflow_hub as hub



def roberta_semantic_algorithm(scrapped_data):

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

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

    return semantic_articles


