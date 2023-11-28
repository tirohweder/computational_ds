import streamlit as st
import pandas as pd
import numpy as np
import os

# Function to load data from a CSV file
def load_data():
    path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output', 'recommender_big_clusters_with_data.csv')
    df = pd.read_csv(path)
    return df

# Function to extract text from the dataframe based on the actual news headline
def extract_text(df, actual_news):
    row = df.loc[df['Headline'] == actual_news].iloc[0]
    link = row['Link'] if row['Link'][:4] == 'http' else row['Organization']
    text = row['Text'][:400] if len(row['Text']) > 400 else row['Text']
    return f"{text}...[read more]({link})"

# Function to generate recommendations based on the actual news headline
def generate_recommendations(actual_news, df):
    cluster = df.loc[df['Headline'] == actual_news].iloc[0]['Cluster_x']
    filtered_df = df[df['Cluster_x'] == cluster]
    results = []
    for i in range(3):
        idx = filtered_df.loc[df['Headline'] == actual_news].iloc[0][str(i)]
        recommendation = filtered_df['Headline'].iloc[idx]
        results.append(recommendation)
    return results

# Function to load DBSCAN plots from HTML files
def load_dbscan():
    path = os.path.join(os.path.dirname(__file__), 'templates')
    # Replace this with your actual code to generate plots
    with open(os.path.join(path, '3d_plot_word2vec.html'), "r", encoding='utf-8') as file:
        fig1 = file.read()

    with open(os.path.join(path, '3d_plot_tfidf.html'), "r", encoding='utf-8') as file:
        fig2 = file.read()

    return fig1, fig2

# Function to load LDA plot from an HTML file
def load_lda():
    path = os.path.join(os.path.dirname(__file__), 'templates')
    # Replace this with your actual code to generate plots
    with open(os.path.join(path, 'best_lda_visualization_3.html'), "r", encoding='utf-8') as file:
        fig1 = file.read()

    return fig1

# Streamlit app
def main():
    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = load_data()
        print('df initialized')
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
        print('index initialized')
    if "actual_news" not in st.session_state:
        st.session_state.actual_news = np.random.choice(st.session_state.df['Headline'])
        # st.session_state.actual_news = 'At least 14 injured in blast, fire in Minneapolis apartment building'
        print('news initialized')
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = generate_recommendations(st.session_state.actual_news, st.session_state.df)
        print('recom initialized')

    # Adjust the width 
    st.set_page_config(layout="wide")

    # Display the title
    st.markdown("<h1 style='text-align: center; color: white;'>Computational Tools for Data Science - Final Project</h2>", unsafe_allow_html=True)

    # Create a two-column layout
    col1, col2 = st.columns([3,2])

    with col1:
        # Function to update the current index based on the arrow button click
        def update_index(direction):
            if direction == "next":
                st.session_state.current_index = (st.session_state.current_index + 1) % 2
            elif direction == "prev":
                st.session_state.current_index = (st.session_state.current_index - 1) % 2

        st.markdown("## Graphs")

        # Display the current image
        st.components.v1.html(load_dbscan()[st.session_state.current_index], height=600, width=800)


    with col2:     
        # Create a section header
        section = st.header("")
        section.header("Results")

        # Display the current news headline
        section.subheader(st.session_state.actual_news)

        # Display the extracted text for the current news headline
        st.write(extract_text(st.session_state.df, st.session_state.actual_news))

        # Add a divider for visual separation
        st.divider()

        # Create buttons for the recommended news headlines
        recom0 = st.button(st.session_state.recommendations[0])
        recom1 = st.button(st.session_state.recommendations[1])
        recom2 = st.button(st.session_state.recommendations[2])

        # Handle button clicks
        if recom0:
            print('recom0')
            st.session_state.actual_news = st.session_state.recommendations[0] 
            section.subheader(st.session_state.actual_news)
            st.session_state.recommendations = generate_recommendations(st.session_state.actual_news, st.session_state.df)
            st.rerun()
        if recom1:
            print('recom1')
            st.session_state.actual_news = st.session_state.recommendations[1] 
            section.subheader(st.session_state.actual_news)
            st.session_state.recommendations = generate_recommendations(st.session_state.actual_news, st.session_state.df)
            st.rerun()
        if recom2:
            print('recom2')
            st.session_state.actual_news = st.session_state.recommendations[2] 
            section.subheader(st.session_state.actual_news)
            st.session_state.recommendations = generate_recommendations(st.session_state.actual_news, st.session_state.df)
            st.rerun()

        st.divider()
    
    st.components.v1.html(load_lda(), height=800)

    # Create a three-column layout
    col3, _, col4 = st.columns([1, 10, 1])

    # Buttons to navigate between DBSCAN plots
    if col3.button("← Previous"):
        update_index("prev")

    if col4.button("Next →"):
        update_index("next")

if __name__ == "__main__":
    main()
