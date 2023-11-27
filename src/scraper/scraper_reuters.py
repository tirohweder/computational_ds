import pandas as pd
import requests
from bs4 import BeautifulSoup

# Read a CSV file into a DataFrame
df = pd.read_csv('../../data/source/reuters_2019.csv', header=None)

# Extract URLs from the DataFrame
urls = df[2].tolist()
urls = urls[1:]

# Define headers for the HTTP requests to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}

scraped_data = []  # List to store scraped data
count = 0  # Counter to track the number of processed URLs

# Loop through each URL and scrape data
for url in urls:
    count += 1
    response = requests.get(url, headers=headers)  # Make an HTTP GET request
    soup = BeautifulSoup(response.content, 'html.parser')  # Parse the response content with BeautifulSoup

    # Find the headline element and extract text
    headline_tag = soup.find('h1', {'class': ['Headline-headline-2FXIq', 'Headline-black-OogpV', 'ArticleHeader-headline-NlAqj']})
    headline_text = headline_tag.text.strip() if headline_tag else None

    # Find the publication date and extract content
    pub_date_meta = soup.find('meta', attrs={"property": "og:article:published_time"})
    pub_date = pub_date_meta['content'] if pub_date_meta else None

    # Find the article element and extract all paragraph texts
    article_element = soup.find('article', {'class': 'ArticlePage-article-body-1xN5M'})
    paragraphs = article_element.find_all('p', {'class': 'Paragraph-paragraph-2Bgue ArticleBody-para-TD_9x'})
    article_content = ' '.join([p.text for p in paragraphs])  # Join paragraph texts

    print(pub_date, headline_text, article_content)  # Printing scraped data

    # Append scraped data to the list
    scraped_data.append([count, headline_text, pub_date, article_content, url, 'Reuters'])

    # Save data to a CSV file every 1000 articles
    if count % 1000 == 0:
        temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date', 'Text', 'Organization', 'Link'])
        with open('data/output/reuters_2019_output.csv', 'a', encoding='utf-8') as f:
            temp_df.to_csv(f, header=f.tell() == 0, index=False, lineterminator='\n')  # Write header only if the file is empty
        scraped_data = []

# Save any remaining scraped data to the CSV file
if scraped_data:
    temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date', 'Text', 'Organization', 'Link'])
    with open('data/output/reuters_2019_output.csv', 'a', encoding='utf-8') as f:
        temp_df.to_csv(f, header=f.tell() == 0, index=False, lineterminator='\n')
