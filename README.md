# Project Computational Data Science

## Scraping the data
Starting of from the start of the project, the first thing to do is run scraper_selection.py this will select which
files to scrape from the ones that much more data than cnn. So it breaks it down. This is already done and the correct
files are then added to data/source with an appended column "selected" with a value of 1.

Running the scraper will create a file in data/output called xxx_year_output.csv
This is the file that contains the data that is used in the analysis and is based in the files from data/source.
There are 3 scraper files, reuters, cnn, and fox. Each is individual and needs to be run for every file.

We will include a zip with all of the data scraped already

# Topic Modeling LDA
This Jupyter Notebook contains code for performing topic modeling using Latent Dirichlet Allocation (LDA) on a text dataset. The code utilizes the gensim library for LDA modeling and pyLDAvis for visualization.
## Prerequisites
To run the code you need to load the CSV file "updated_dataframe_with_clusters_word2vec.csv" or the scraped files located in _src/data/output_

Make sure you have the following Python packages installed:
- gensim
- pandas
- nltk
- pyLDAvis

    ```bash
        pip install gensim pandas nltk pyLDAvis
        ```

## Results
- **coherence_scores.csv**: CSV file containing coherence scores for different numbers of topics.
- **document_topic_probabilities.csv**: CSV file containing document-topic probabilities for the selected LDA model.
-**best_lda_visualization.html**: HTML file containing the visualization of the best LDA model.
 
# Topic Modeling Vectorizer
Prerequisites: in _src/data/output_ the scraped files need to be present.
For Word2Vec, the Model _GoogleNews-vectors-negative300.bin_ needs to be present in _src/topic_modeling/vectorization_
As the computation of the vectorization and for tfidf takes a long time, find the precomputed files which can be loaded by the code inside of the attached zip file. Just place the files in _src/topic_modeling/vectorization_ 

Word2Vec:
word2vec_embeddings.npy - the complete data as word2vec embeddings
umap_embeddings_word2vec.npy - the umap reduction of the word2vec embeddings
hdbscan_cluster_labels_word2vec_15.npy - after hdbscan clustering


Word2Vec Generated files:
minimum_spanning_tree_visualization_word2vec_15.png - the minimum spanning tree of the hdbscan clusters
centroid_distance_matrix_word2vec_15.csv - the distance matrix of the centroids of the hdbscan clusters
updated_dataframe_with_clusters_word2vec_15.csv - the dataframe with the hdbscan clusters added as a column
3d_plot_word2vec_15.html - the 3d plot of the hdbscan clusters

TFIDF:
The same but replaced word2vec with tfidf

Changing the min_cluster_size can be done in this line: hdbscan.HDBSCAN(**min_cluster_size=15**,...)

The evaluation of the clusters is done in the _evaluation_cluster_similarity.py_

# Sentiment Analysis


# Recommender System

# Graph Visualisation
For the visualisation to work, we need two additional files in _src/graph_visualisation_:
_centroid_distance_matrix_word2vec_15.csv_ 
_ines_add_this_please_ - please find this attached in the additional zip




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


