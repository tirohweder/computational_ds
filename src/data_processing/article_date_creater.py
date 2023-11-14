# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:18:12 2023

@author: storr
"""
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

path_cnn_2014 = r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\CNN_2014_2016\cnn_scraped_2014.csv'
path_cnn_2015 = r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\CNN_2014_2016\cnn_scraped_2015.csv'
path_cnn_2016 = r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\CNN_2014_2016\cnn_scraped_2016.csv'
path_cnn_2017 = r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\cnn_2017_scrapped.csv'
path_cnn_2018 = r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\cnn_2018_scrapped.csv'
path_cnn_2019 = r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\cnn_2019_scrapped.csv'

data_cnn_2014 = pd.read_csv(path_cnn_2014)
data_cnn_2015 = pd.read_csv(path_cnn_2015)
data_cnn_2016 = pd.read_csv(path_cnn_2016)
data_cnn_2017 = pd.read_csv(path_cnn_2017)
data_cnn_2018 = pd.read_csv(path_cnn_2018)
data_cnn_2019 = pd.read_csv(path_cnn_2019)


datas = [data_cnn_2014,data_cnn_2015,data_cnn_2016,data_cnn_2017,data_cnn_2018,data_cnn_2019]
plt.figure(figsize=(10, 6))

weekly_articles_per_year = []
weekly_percentage_year = []
    
for data in datas:   
            
    data.dropna(subset=['Date'], inplace=True)
    
            
    data['Clean Dates'] = data['Date'].apply(lambda x: ' '.join(x.split()[-3:]))

    data['Clean Dates'] = pd.to_datetime(data['Clean Dates'], format='%B %d, %Y')

    
    data['Week'] = data['Clean Dates'].dt.week
    
    
    weekly_counts = data['Week'].value_counts().sort_index()
    weekly_counts_percentage = weekly_counts/sum(weekly_counts)
    
    
    weekly_counts_percentage.plot(kind='line', label = data['Clean Dates'].dt.year.iloc[0])

    weekly_articles_per_year.append(weekly_counts[0:52])
    weekly_percentage_year.append(weekly_counts_percentage[0:52])

plt.title('Articles in each week')
plt.legend()

plt.show    

weekly_articles_per_year = np.array(weekly_articles_per_year).T
weekly_percentage_year = np.array(weekly_percentage_year).T
weekly_articles_avg = np.mean(weekly_articles_per_year, axis = 0).T

plt.plot(weekly_articles_avg, label = 'Avg')


plt.title('Articles in each week')
plt.legend()
plt.xlabel('Week')
plt.ylabel('Articles')
plt.grid(axis='y')
plt.show 


weekly_data = np.concatenate((weekly_articles_per_year,weekly_percentage_year), axis=1)
weekly_articles_data = pd.DataFrame(weekly_data, columns=['2014','2015','2016','2017','2018','2019','2014 %','2015 %','2016 %','2017 %','2018 %','2019 %'])

#weekly_articles_data.to_csv(r'C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 1\Semester 1\Computational Tools for Data Science\Project\DATA\cnn_weekly_data.csv', index=False)

    