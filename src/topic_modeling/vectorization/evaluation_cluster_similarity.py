import pandas as pd
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
import itertools
import matplotlib.pyplot as plt
import numpy as np

def calculate_all_metrics(df1, df2):
    ari = adjusted_rand_score(df1['Cluster'], df2['Cluster'])
    fmi = fowlkes_mallows_score(df1['Cluster'], df2['Cluster'])
    nmi = normalized_mutual_info_score(df1['Cluster'], df2['Cluster'])
    return ari, fmi, nmi

def calculate_descriptive_stats(df):
    total_clusters = df['Cluster'].nunique()
    no_cluster_assignments = (df['Cluster'] == -1).sum()
    assigned_clusters = df[df['Cluster'] != -1]
    average_cluster_size = assigned_clusters['Cluster'].value_counts().mean()
    median_cluster_size = assigned_clusters['Cluster'].value_counts().median()
    max_cluster_size = assigned_clusters['Cluster'].value_counts().max()
    min_cluster_size = assigned_clusters['Cluster'].value_counts().min()

    return {
        'Total Clusters': total_clusters,
        'No Cluster Assignments': no_cluster_assignments,
        'Average Cluster Size': average_cluster_size,
        'Median Cluster Size': median_cluster_size,
        'Max Cluster Size': max_cluster_size,
        'Min Cluster Size': min_cluster_size
    }

def plot_metrics_comparison(all_results):
    labels = list(all_results.keys())
    ari_scores = [result[0] for result in all_results.values()]
    fmi_scores = [result[1] for result in all_results.values()]
    nmi_scores = [result[2] for result in all_results.values()]

    print(f'ARI Scores: {ari_scores}')
    print(f'FMI Scores: {fmi_scores}')
    print(f'NMI Scores: {nmi_scores}')
    bar_width = 0.25
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(10, 6))
    plt.bar(r1, ari_scores, color='#0b132b', width=bar_width, edgecolor='grey', label='ARI')
    plt.bar(r2, fmi_scores, color='#1c2541', width=bar_width, edgecolor='grey', label='FMI')
    plt.bar(r3, nmi_scores, color='#3a506b', width=bar_width, edgecolor='grey', label='NMI')

    plt.xlabel('Dataset Pairs', fontweight='bold', fontsize=15)
    plt.ylabel('Scores', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width for r in range(len(labels))], labels)
    plt.legend()
    plt.title('Comparison of Dataset Pairs using ARI, FMI, and NMI')
    plt.show()

# Load the datasets
file_paths = [
    r"C:\Users\rohwe\Documents\DMAI\computational_ds\src\topic_modeling\vectorization\old"
    r"\updated_dataframe_with_clusters_word2vec_15.csv",
    r"C:\Users\rohwe\Documents\DMAI\computational_ds\src\topic_modeling\vectorization\old"
    r"\updated_dataframe_with_clusters_word2vec_30.csv",
    r"C:\Users\rohwe\Documents\DMAI\computational_ds\src\topic_modeling\vectorization\old"
    r"\updated_dataframe_with_clusters_word2vec_50.csv",
    r"C:\Users\rohwe\Documents\DMAI\computational_ds\src\topic_modeling\vectorization"
    r"\updated_dataframe_with_clusters_tfidf_15.csv",

]
#    "updated_dataframe_with_clusters_word2vec_50.csv"

# Reading the datasets into dataframes
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Calculate metrics and descriptive statistics for each dataset pair
all_results = {}
descriptive_stats = {}
pair_labels = ['15-30', '15-50', '30-50']

for (i, j), label in zip(itertools.combinations(range(len(file_paths)), 2), pair_labels):
    all_results[label] = calculate_all_metrics(dfs[i], dfs[j])
    descriptive_stats[f'Dataset {i+1}'] = calculate_descriptive_stats(dfs[i])
    descriptive_stats[f'Dataset {j+1}'] = calculate_descriptive_stats(dfs[j])

# Plotting the metrics comparison
plot_metrics_comparison(all_results)

# Optional: Print descriptive statistics
for dataset, stats in descriptive_stats.items():
    print(f"{dataset}:\n", "\n".join([f"{key}: {value}" for key, value in stats.items()]), "\n")
