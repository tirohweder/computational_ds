import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# Load the CSV file
df = pd.read_csv('data/source/fox_2018.csv', header=None)

urls = df[2].tolist()
urls = urls[1:]

# Set the batch size
batch_size = 1000
total_links = len(urls)
start_index = 0 

count = 0 

output_file = 'output_fox_2018.csv'

while start_index < total_links:
    end_index = min(start_index + batch_size, total_links)
    batch_urls = urls[start_index:end_index]

    scraped_data = []

    for url in batch_urls:  
        count += 1
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
    
        headline_tag = soup.find('h1', {'class': 'headline'})
        if headline_tag:
            headline_text = headline_tag.text.strip()
        else:
            headline_text = None

        timestamp_tag = soup.find('span', {'class': 'article-date'})
        if timestamp_tag:
            timestamp = ' '.join(timestamp_tag.text.replace('\n', ' ').replace('\t', ' ').strip().split())
        else:
            timestamp = None

        # Parsing article content
        article_content_tag = soup.find('div', {'class': 'article-body'})
        if article_content_tag:
            article_content = ' '.join(article_content_tag.text.replace('\n', ' ').replace('\t', ' ').split())
        else:
            article_content = None
        
        scraped_data.append([count, headline_text, timestamp, article_content, url, 'FOX'])
        

    output_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date', 'Text', 'Link', 'Organization'])

    if os.path.exists(output_file):
        output_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        output_df.to_csv(output_file, index=False)

    # Increment the start_index to continue with the next batch
    start_index = end_index
