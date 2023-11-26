import pandas as pd
import numpy as np

# cnn_weekly_ref_data.csv is the file that contains the distribution of articles for each week for cnn and is the
# minimum
distribution_df = pd.read_csv('../../data/other/cnn_weekly_ref_data.csv')

for year in range(2014, 2020):
    # Change for "fox" to "cnn" and "reuters" as needed
    source_df = pd.read_csv(f'../../data/source/fox_{year}.csv')
    source_df['selected'] = 0

    # Keep track of the cumulative percentage
    cumulative_percentage = 0

    # Loop over each week
    for index, row in distribution_df.iterrows():
        # Determine the percentage range for this week
        start_percentage = cumulative_percentage
        end_percentage = cumulative_percentage + row[f'{year} %']
        cumulative_percentage = end_percentage

        # Calculate the index range for this week
        start_index = int(round(start_percentage * len(source_df)))
        end_index = int(round(end_percentage * len(source_df)))

        # Get the number of articles to select for this week
        number_to_select = int(row[str(year)])
        number_to_select = min(number_to_select, end_index - start_index)

        if number_to_select > 0:
            selected_indices = np.random.choice(source_df.index[start_index:end_index], number_to_select, replace=False)
            source_df.loc[selected_indices, 'selected'] = 1


    source_df.to_csv(f'../../data/source/fox_{year}.csv', index=False)
