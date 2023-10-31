import csv
from io import StringIO
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the CSV file
df1 = pd.read_csv('../data/source/cnn_2017.csv', header=None)

df2 = pd.read_csv('../data/output/cnn_2017_output.csv', header=None)

# Get the URLs
urls_1 = df1[2].tolist()
urls_2 = df2[4].tolist()


unique_df1 = df1[~df1[2].isin(df2[4])]

unique_df1.to_csv('unique_df1.csv', index=False, header=False)

