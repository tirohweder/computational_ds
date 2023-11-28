import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the CSV file containing URLs.
df = pd.read_csv('../../data/source/cnn_2017.csv', header=None)

# Extract the list of URLs from the dataframe.
urls = df[2].tolist()

# Initialize an empty list to store the scraped data.
scraped_data = []
count = 0

# Iterate over each URL to scrape data.
for url in urls:
    count += 1
    # Make a request to the URL and create a BeautifulSoup object.
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Scrape the headline text, if available.
    headline_tag = soup.find('h1', {'class': 'headline__text'})
    if headline_tag:
        headline_text = headline_tag.text.strip()
    else:
        headline_text = None

    # Scrape the timestamp, if available.
    timestamp_tag = soup.find('div', {'class': 'timestamp'})
    if timestamp_tag:
        timestamp = ' '.join(timestamp_tag.text.replace('\n', ' ').replace('\t', ' ').strip().split())
    else:
        timestamp = None

    # Scrape the article content, if available.
    article_content_tag = soup.find('div', {'class': 'article__content'})
    if article_content_tag:
        article_content = ' '.join(article_content_tag.text.replace('\n', ' ').replace('\t', ' ').split())
    else:
        article_content = None

    # Append the scraped data to the list.
    scraped_data.append([count, headline_text, timestamp, article_content, url, 'CNN'])

    # Save the scraped data to a CSV file every 1000 iterations.
    if count % 1000 == 0:
        temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date', 'Text', 'Organization', 'Link'])
        with open('../../data/output/cnn_2017_output.csv', 'a', encoding='utf-8') as f:
            temp_df.to_csv(f, header=f.tell() == 0, index=False, lineterminator='\n')
        scraped_data = []

# Save any remaining scraped data that wasn't saved in the loop.
if scraped_data:
    temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date', 'Text', 'Organization', 'Link'])
    with open('../../data/output/cnn_2017_output.csv', 'a', encoding='utf-8') as f:
        temp_df.to_csv(f, header=f.tell() == 0, index=False, lineterminator='\n')
