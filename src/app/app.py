import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

def load_data():
    path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output', 'updated_dataframe_with_clusters_word2vec.csv')
    df = pd.read_csv(path)
    return df

def extract_text(df, actual_news):
    row = df.loc[df['Headline'] == actual_news].iloc[0]
    link = row['Link']
    text = row['Text'][:400] if len(row['Text']) > 400 else row['Text']
    return f"{text}...[read more]({link})"

# Your code to generate results
def generate_results(df):
    # Replace this with your actual code to generate results
    results = np.random.choice(df['Headline'], 3)
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
    # Adjust the width 
    st.set_page_config(layout="wide")

    st.markdown("<h1 style='text-align: center; color: white;'>Computational Tools for Data Science - Final Project</h2>", unsafe_allow_html=True)

    # Create a two-column layout
    col1, col2 = st.columns([3,2])

    with col1:
        # Function to update the current index based on the arrow button click
        def update_index(direction):
            global current_index
            if direction == "next":
                current_index = (current_index + 1) % 2
            elif direction == "prev":
                current_index = (current_index - 1) % 2

        st.markdown("## Graphs")

        # Display the current image
        st.components.v1.html(load_dbscan()[current_index], height=600, width=800, scrolling=True)

        st.components.v1.html(load_lda()[current_index], height=600, width=800, scrolling=True)

        if st.button("← Previous"):
            update_index("prev")

        if st.button("Next →"):
            update_index("next")

    with col2:
        global actual_news, recommendations
        # # Display results in the right column
        st.header("Results")
        selected_text = st.header("Select a text:")
        selected_text.subheader(actual_news)
        st.write(extract_text(df, actual_news))
        st.divider()
        for text in recommendations:
            if st.button(text):
                recommendations = generate_results(df)
                actual_news = text
                selected_text.text(f"Selected Text: {text}")
                break
        st.divider()
        


df = load_data()

# Generate and display results from your code
actual_news = np.random.choice(df['Headline'])
recommendations = generate_results(df)

current_index = 0

if __name__ == "__main__":
    main()
