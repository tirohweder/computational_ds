# Project Computational Data Science

## Scraping the data
Starting of from the start of the project, the first thing to do is run scraper_selection.py this will select which
files to scrape from the ones that much more data than cnn. So it breaks it down. This is already done and the correct
files are then added to data/source with an appended column "selected" with a value of 1.

Running the scraper will create a file in data/output called xxx_year_output.csv
This is the file that contains the data that is used in the analysis and is based in the files from data/source.
There are 3 scraper files, reuters, cnn, and fox. Each is individual and needs to be run for every file.



# Topic Modeling LDA

# Topic Modeling Vectorizer


# Sentiment Analysis


# Recommender System

# Graph Visualisation


# Website Visualisation

This Streamlit app is designed for visualizing and exploring the results of our project. It includes features such as the recommender system for recommending related news articles and visualizing the data using DBSCAN and LDA plots.

## Setup

1. Make sure you have Python installed on your machine.

2. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run src/app/app.py
    ```

4. Open your web browser and navigate to the provided URL.

## Usage

Once the app is running, you can explore the following features:

- **Graphs Section:** Navigate through DBSCAN and LDA plots using the arrow buttons to the left. The plots are interactive, and provide insights into the clustering.

- **Results Section:** View the current news headline along with extracted text. Some recommendations are shown below the article. Clicking on a recommendation reloads the page and displays the text of the selected news article.

- **Navigation Buttons:** Use the "← Previous" and "Next →" buttons to navigate between DBSCAN plots.

## Streamlit Cloud Deployment

You can also access the app on Streamlit Cloud. Visit the following URL to view the app:

[Streamlit Cloud App](https://project-datatools.streamlit.app/)

## Dependencies

- `streamlit`: The main framework for building the web app.
- `pandas`, `numpy`: Handling and processing data.
- `os`: Interacting with the operating system for file paths.
- Other dependencies as specified in the `requirements.txt` file.

## Data

The app loads data from the CSV file: `recommender_big_clusters_with_data.csv`. Make sure this file is present in the specified directory '/data/output'.


