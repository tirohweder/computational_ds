import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# Read the CSV file containing URLs
df = pd.read_csv('../../data/source/fox_2014.csv', header=None)

# Filter the DataFrame to include only selected articles
filtered_df = df[df[3] == "1"]

# Extract URLs from the filtered DataFrame
urls = filtered_df[2].tolist()
urls = urls[1:]

# Set the number of URLs to process in each batch to prevent data loss in case of a crash
batch_size = 1000
total_links = len(urls)
start_index = 0

count = 0  # Initialize a counter for scraped articles

# Define the output file path for the scraped data
output_file = '../../output/fox_2016.csv'

# Loop over batches of URLs
while start_index < total_links:
    end_index = min(start_index + batch_size, total_links)  # Determine the end index for the current batch
    batch_urls = urls[start_index:end_index]  # Extract the current batch of URLs

    scraped_data = []  # List to store data for the current batch

    for url in batch_urls:
        count += 1
        response = requests.get(url)  # Send a request to the URL
        soup = BeautifulSoup(response.content, 'html.parser')  # Parse the HTML content

        # Scrape the headline
        headline_tag = soup.find('h1', {'class': 'headline'})
        headline_text = headline_tag.text.strip() if headline_tag else None

        # Scrape the timestamp
        timestamp_tag = soup.find('span', {'class': 'article-date'})
        timestamp = ' '.join(
            timestamp_tag.text.replace('\n', ' ').replace('\t', ' ').strip().split()) if timestamp_tag else None

        # Scrape the article content
        article_content_tag = soup.find('div', {'class': 'article-body'})
        article_content = ' '.join(
            article_content_tag.text.replace('\n', ' ').replace('\t', ' ').split()) if article_content_tag else None

        # Append the scraped data to the list
        scraped_data.append([count, headline_text, timestamp, article_content, 'FOX', url])

    # Create a DataFrame from the scraped data
    output_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date', 'Text', 'Organization', 'Link'])

    # Save the DataFrame to a CSV file, appending to it if it already exists
    if os.path.exists(output_file):
        output_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        output_df.to_csv(output_file, index=False)

    # Update the start_index for the next batch
    start_index = end_index
