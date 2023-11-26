import pandas as pd
import requests
from bs4 import BeautifulSoup

# Change this for each file
df = pd.read_csv('../../data/source/reuters_2019.csv', header=None)

urls = df[2].tolist()
urls = urls[1:]
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}

scraped_data = []
count = 0
for url in urls:
    count+= 1
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    headline_tag = soup.find('h1', {'class': ['Headline-headline-2FXIq', 'Headline-black-OogpV', 'ArticleHeader-headline-NlAqj']})

    if headline_tag:
        headline_text = headline_tag.text.strip()
    else:
        headline_text = None

    pub_date_meta = soup.find('meta', attrs={"property": "og:article:published_time"})
    if pub_date_meta:
        pub_date = pub_date_meta['content']

    article_element = soup.find('article', {'class': 'ArticlePage-article-body-1xN5M'})

    # Extract all paragraph texts within the article
    paragraphs = article_element.find_all('p', {'class': 'Paragraph-paragraph-2Bgue ArticleBody-para-TD_9x'})

    # Join the paragraphs to form the full article content
    article_content = ' '.join([p.text for p in paragraphs])


    print (pub_date, headline_text, article_content)

    scraped_data.append([count, headline_text, pub_date , article_content,url, 'Reuters'])
    if count % 1000 == 0:
        temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date',  'Text', 'Organization', 'Link'])
        with open('data/output/reuters_2019_output.csv', 'a' , encoding='utf-8') as f:
            temp_df.to_csv(f, header=f.tell()==0, index=False,  lineterminator='\n')  # Write header only if file is empty
        scraped_data = []


if scraped_data:
    temp_df = pd.DataFrame(scraped_data, columns=['ID', 'Headline', 'Date',  'Text', 'Organization', 'Link'])
    with open('data/output/reuters_2019_output.csv', 'a', encoding='utf-8') as f:
        temp_df.to_csv(f, header=f.tell()==0, index=False, lineterminator='\n')