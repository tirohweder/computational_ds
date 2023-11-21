import numpy as np
import pandas as pd
import sys
import os
sys.path.append('../src')
sys.path.append('../data')
import  semantic_analysis

base_path = r"C:\Users\inest\OneDrive - Danmarks Tekniske Universitet\Semester I\Computational Tools for Data Science\data"

file_path = os.path.join(base_path, f"reuters_2014_output.csv")

if os.path.exists(file_path):
    #print(semantic_analysis.lexicon_nltk(file_path))

    df = pd.read_csv(file_path)
    df_2 = pd.read_csv(os.path.join(base_path, f"reuters_2014_semantic.csv"))


    rob_values = df_2[[df_2'Semantic roberta twitter']== 'neutral']
    values = df[(df['Sentiment_Lexicon'] < 0.25) & (df['Sentiment_Lexicon'] > -0.75)]

    #print(rob_values)
    #print(values)

else:
    print(f"No file found for year {2014}")


