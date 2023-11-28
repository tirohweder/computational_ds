import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams



def sentiment_count_graph(master_file_df):

    rcParams['font.family'] = 'Times New Roman'

    cnn_master = master_file_df[master_file_df['Organization']=='CNN']
    cnn_roberta_sentiments_counts = cnn_master['Semantic roberta twitter'].value_counts() / len(cnn_master)*100
 
    fox_master = master_file_df[master_file_df['Organization']=='FOX']
    fox_roberta_sentiments_counts = fox_master['Semantic roberta twitter'].value_counts() / (len(fox_master))*100

    reuters_master = master_file_df[master_file_df['Organization']=='Reuters']
    reuters_roberta_sentiments_counts = reuters_master['Semantic roberta twitter'].value_counts() / len(reuters_master)*100

    sentiments = ['negative', 'neutral', 'positive']

    bar_width = 0.2
    index = range(len(sentiments))

    colors = ['#FF6666', '#66B2FF', '#99FF99'] 

    for j, sentiment, color in zip(index, sentiments, colors):
        plt.bar([i + j*bar_width for i in index], [cnn_roberta_sentiments_counts[sentiment],
                                                   fox_roberta_sentiments_counts[sentiment],
                                                   reuters_roberta_sentiments_counts[sentiment]],
                bar_width, label=sentiment, color=color)

    plt.xlabel('Sentiments')
    plt.ylabel('Relative weight [%]')
    plt.title('Sentiment Analysis by Organization')
    plt.xticks([i + bar_width for i in index], ['CNN', 'FOX', 'Reuters'])
    plt.legend()

    plt.gcf().set_size_inches(4, 2) 

    plt.show()