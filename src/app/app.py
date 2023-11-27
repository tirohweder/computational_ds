import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

def load_data():
    path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output', 'recommender_big_clusters_with_data.csv')
    df = pd.read_csv(path)
    return df

def extract_text(df, actual_news):
    row = df.loc[df['Headline'] == actual_news].iloc[0]
    link = row['Link']
    text = row['Text'][:400] if len(row['Text']) > 400 else row['Text']
    return f"{text}...[read more]({link})"

# Your code to generate results
def generate_results(actual_news, df):
    # Replace this with your actual code to generate results
    cluster = df.loc[df['Headline'] == actual_news].iloc[0]['Cluster']
    filtered_df = df[df['Cluster'] == cluster]
    results = np.random.choice(filtered_df['Headline'], 3)
    return results

# Your code to generate Plotly plots
def load_dbscan():
    path = os.path.join(os.path.dirname(__file__), 'templates')
    # Replace this with your actual code to generate plots
    with open(os.path.join(path, '3d_plot_word2vec.html'), "r", encoding='utf-8') as file:
        fig1 = file.read()

    with open(os.path.join(path, '3d_plot_tfidf.html'), "r", encoding='utf-8') as file:
        fig2 = file.read()

    return fig1, fig2

def load_lda():
    path = os.path.join(os.path.dirname(__file__), 'templates')
    # Replace this with your actual code to generate plots
    with open(os.path.join(path, 'best_lda_visualization.html'), "r", encoding='utf-8') as file:
        fig1 = file.read()

    with open(os.path.join(path, 'lda_visualization.html'), "r", encoding='utf-8') as file:
        fig2 = file.read()

    return fig1, fig2

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
        st.session_state.recommendations = generate_results(st.session_state.actual_news, st.session_state.df)
        print('recom initialized')

    # Adjust the width 
    st.set_page_config(layout="wide")

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
        section = st.header("")
        section.header("Results")
        section.subheader(st.session_state.actual_news)
        # st.write(extract_text(st.session_state.df, st.session_state.actual_news))
        st.divider()
        recom0 = st.button(st.session_state.recommendations[0])
        recom1 = st.button(st.session_state.recommendations[1])
        recom2 = st.button(st.session_state.recommendations[2])
        if recom0:
            print('recom0')
            st.session_state.actual_news = st.session_state.recommendations[0] 
            section.subheader(st.session_state.actual_news)
            st.session_state.recommendations = generate_results(st.session_state.actual_news, st.session_state.df)
            st.rerun()


        if recom1:
            print('recom1')
            st.session_state.actual_news = st.session_state.recommendations[1] 
            section.subheader(st.session_state.actual_news)
            st.session_state.recommendations = generate_results(st.session_state.actual_news, st.session_state.df)
            st.rerun()


        if recom2:
            print('recom2')
            st.session_state.actual_news = st.session_state.recommendations[2] 
            section.subheader(st.session_state.actual_news)
            st.session_state.recommendations = generate_results(st.session_state.actual_news, st.session_state.df)
            st.rerun()


        st.divider()
    
    st.components.v1.html(load_lda()[st.session_state.current_index], height=800)

    # Create a two-column layout
    col3, _, col4 = st.columns([1, 10, 1])

    if col3.button("← Previous"):
        update_index("prev")

    if col4.button("Next →"):
        update_index("next")

if __name__ == "__main__":
    main()
