import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the CSV file
df = pd.read_csv('data/source/cnn_2016.csv', header=None)

# Get the URLs
urls = df[2].tolist()
urls = urls[1:]

scraped_data = []
count = 0
for url in urls:
    count+= 1
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    headline_tag = soup.find('h1', {'class': 'headline__text'})
    if headline_tag:
        headline_text = headline_tag.text.strip()
    else:
        headline_text = None

    # Parsing timestamp
    timestamp_tag = soup.find('div', {'class': 'timestamp'})
    if timestamp_tag:
        timestamp = ' '.join(timestamp_tag.text.replace('\n', ' ').replace('\t', ' ').strip().split())
    else:
        timestamp = None

    # Parsing article content
    article_content_tag = soup.find('div', {'class': 'article__content'})
    if article_content_tag:
        article_content = ' '.join(article_content_tag.text.replace('\n', ' ').replace('\t', ' ').split())
    else:
        article_content = None

    print (headline_text, timestamp, article_content)

    scraped_data.append([count, headline_text, timestamp , article_content,url, 'CNN'])
    if count % 1000 == 0:
        temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date',  'Text', 'Organization', 'Link'])
        with open('data/output/cnn_2016_output.csv', 'a' , encoding='utf-8') as f:
            temp_df.to_csv(f, header=f.tell()==0, index=False,  lineterminator='\n')  # Write header only if file is empty
        scraped_data = []
# Create a DataFrame from the scraped data


if scraped_data:
    temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date',  'Text', 'Organization', 'Link'])
    with open('data/output/cnn_2016_output.csv', 'a', encoding='utf-8') as f:
        temp_df.to_csv(f, header=f.tell()==0, index=False, lineterminator='\n')