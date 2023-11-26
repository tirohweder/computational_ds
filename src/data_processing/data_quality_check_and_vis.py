import pandas as pd
import os

# Define the directory where the files are stored
directory = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output')

def overview_source():
    summary_dict = {}

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and ('cnn_' in filename or 'fox_' in filename or 'reuters_' in filename):
            # Extract the year and source from the filename
            parts = filename.split('_')
            source = parts[0]
            year = parts[1].split('.')[0]

            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Count the number of rows
            row_count = df.shape[0]

            # Add the count to the summary dictionary
            summary_dict[(source, year)] = summary_dict.get((source, year), 0) + row_count

    # Convert the dictionary to a DataFrame for better visualization and analysis
    summary_df = pd.DataFrame(list(summary_dict.items()), columns=['Source_Year', 'RowCount'])
    summary_df[['Source', 'Year']] = pd.DataFrame(summary_df['Source_Year'].tolist(), index=summary_df.index)
    summary_df.drop('Source_Year', axis=1, inplace=True)

    # Pivot the DataFrame to have sources as columns and years as rows
    pivot_df = summary_df.pivot(index='Year', columns='Source', values='RowCount')

    # Print the summary pivot table
    print(pivot_df)

    # Optionally, save the pivot table to a CSV file
    pivot_df.to_csv('annual_summary.csv')




def check_data_quality():
    report_dict = {}
    total_rows_across_all_files = 0  # Initialize a counter for total rows

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            try:
                # Construct the full path to the file
                file_path = os.path.join(directory, filename)
                # Load the CSV file
                df = pd.read_csv(file_path)

                # Update total rows count
                total_rows_across_all_files += df.shape[0]

                # Create a report for the current file
                report = {
                    'total_rows': df.shape[0],
                    'total_columns': df.shape[1],
                    'missing_values': df.isnull().sum().to_dict(),
                    'complete_rows': df.dropna().shape[0]
                }
                report_dict[filename] = report

            except Exception as e:
                # If there's an error loading the CSV, add it to the report
                report_dict[filename] = {'error': str(e)}

    # Print the data quality report
    print("\nData Quality Report:")
    for file, report in report_dict.items():
        print(f"{file}: {report}")

    # Print the total number of rows across all files
    print(f"\nTotal number of rows across all files: {total_rows_across_all_files}")



def remove_missing_values():
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            try:
                # Construct the full path to the file
                file_path = os.path.join(directory, filename)
                # Load the CSV file
                df = pd.read_csv(file_path)

                # Remove rows where 'Headline' or 'Text' is missing
                cleaned_df = df.dropna(subset=['Headline', 'Text'])

                # Save the cleaned data back to the file
                cleaned_df.to_csv(file_path, index=False)

                print(f"Processed {filename}: Removed {df.shape[0] - cleaned_df.shape[0]} incomplete rows.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")




overview_source()
#remove_missing_values()
#check_data_quality()
