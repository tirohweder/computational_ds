import pandas as pd
import numpy as np

# Reading a CSV file that contains the distribution of CNN articles by week
distribution_df = pd.read_csv('../../data/other/cnn_weekly_ref_data.csv')

for year in range(2014, 2020):
    # Reading a CSV file for a specific news source (Change fox to cnn or reuters)
    source_df = pd.read_csv(f'../../data/source/fox_{year}.csv')
    # Initializing a column 'selected' to mark selected articles
    source_df['selected'] = 0

    # Variable to keep track of the cumulative distribution percentage
    cumulative_percentage = 0

    # Iterating over each week in the distribution data
    for index, row in distribution_df.iterrows():
        # Calculating the start and end percentage of articles for the current week
        start_percentage = cumulative_percentage
        end_percentage = cumulative_percentage + row[f'{year} %']
        cumulative_percentage = end_percentage

        # Determining the index range in the source DataFrame for the current week
        start_index = int(round(start_percentage * len(source_df)))
        end_index = int(round(end_percentage * len(source_df)))

        # Determining the number of articles to select for this week
        number_to_select = int(row[str(year)])
        # Ensuring the number to select does not exceed the available range
        number_to_select = min(number_to_select, end_index - start_index)

        # Selecting articles randomly within the calculated range
        if number_to_select > 0:
            selected_indices = np.random.choice(source_df.index[start_index:end_index], number_to_select, replace=False)
            source_df.loc[selected_indices, 'selected'] = 1

    # Saving the updated DataFrame back to CSV
    source_df.to_csv(f'../../data/source/fox_{year}.csv', index=False)
