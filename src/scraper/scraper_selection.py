import pandas as pd
import numpy as np

# Load the distribution CSV data
distribution_df = pd.read_csv('../../data/other/cnn_weekly_ref_data.csv')


# Loop over each year
for year in range(2014, 2020):
    # Load the corresponding source file for the entire year
    source_df = pd.read_csv(f'../../data/source/fox_{year}.csv')  # Use the appropriate path and filename
    source_df['selected'] = 0  # Initialize the selected column with 0

    # Keep track of the cumulative percentage
    cumulative_percentage = 0

    # Loop over each week
    for index, row in distribution_df.iterrows():
        # Determine the percentage range for this week
        start_percentage = cumulative_percentage
        end_percentage = cumulative_percentage + row[f'{year} %']
        cumulative_percentage = end_percentage  # Update the cumulative percentage

        # Calculate the index range for this week
        start_index = int(round(start_percentage * len(source_df)))
        end_index = int(round(end_percentage * len(source_df)))

        # Get the number of articles to select for this week
        number_to_select = int(row[str(year)])

        # If there are not enough articles between the calculated indices, adjust the number to select
        number_to_select = min(number_to_select, end_index - start_index)

        # Select the indices for this week
        if number_to_select > 0:
            selected_indices = np.random.choice(source_df.index[start_index:end_index], number_to_select, replace=False)
            source_df.loc[selected_indices, 'selected'] = 1

    # Save the updated source file for the year
    #source_df.to_csv(f'../../data/source/selected_fox_{year}.csv', index=False)
