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
    error_rows_data = []


    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and ('cnn_' in filename or 'fox_' in filename or 'reuters_' in filename):
            try:
                # Construct the full path to the file
                file_path = os.path.join(directory, filename)
                # Load the CSV file
                df = pd.read_csv(file_path)

                # Identify error rows (rows with missing values)
                error_rows = df[df.isnull().any(axis=1)]
                # Add a column to indicate the origin of the file
                error_rows.insert(0, 'origin', filename)

                # Append the error rows to the error_rows_data list
                error_rows_data.append(error_rows)

                # Create a report for the current file
                report = {
                    'total_rows': df.shape[0],
                    'total_columns': df.shape[1],
                    'missing_values': df.isnull().sum().to_dict(),
                    'missing_rows': error_rows.shape[0],
                    'complete_rows': df.dropna().shape[0]
                }
                report_dict[filename] = report

            except Exception as e:
                # If there's an error loading the CSV, add it to the report
                report_dict[filename] = {'error': str(e)}

    # Combine all error rows into a single DataFrame
    if error_rows_data:
        all_error_rows = pd.concat(error_rows_data)
        # Save the error rows to a new CSV file
        all_error_rows.to_csv(r'output_error_rows.csv', index=False)

    # Print the data quality report
    print("\nOutput Directory Data Quality Report:")
    for file, report in report_dict.items():
        print(f"{file}: {report}")


def compare_links():
    # Dictionary to hold the missing links report
    data_source_dir = os.path.join(os.path.dirname(__file__), '../..', 'data', 'source')
    data_output_dir = os.path.join(os.path.dirname(__file__), '../..', 'data', 'output')

    missing_links_report = {}

    # Get the list of source files
    source_files = [f for f in os.listdir(data_source_dir) if f.endswith('.csv')]

    # Loop through each source file
    for source_file in source_files:
        # Construct the corresponding output file name
        base_filename = os.path.splitext(source_file)[0]
        output_file = f"{base_filename}_output.csv"

        # Construct the full paths to the source and output files
        source_file_path = os.path.join(data_source_dir, source_file)
        output_file_path = os.path.join(data_output_dir, output_file)

        # Check if the corresponding output file exists
        if os.path.exists(output_file_path):
            try:
                # Load the data from the source and output files
                source_df = pd.read_csv(source_file_path)
                output_df = pd.read_csv(output_file_path)

                # Get the links from the specified columns
                source_links = source_df.iloc[:, 2].dropna().unique()
                output_links = output_df.iloc[:, 4].dropna().unique()

                # Find the difference in links
                missing_links = set(source_links) - set(output_links)

                # Add the report
                missing_links_report[source_file] = {
                    'missing_links_count': len(missing_links),
                    'missing_links': list(missing_links)
                }

            except Exception as e:
                # Add error to the report if files could not be processed
                missing_links_report[source_file] = {'error': str(e)}
        else:
            # Output file does not exist
            missing_links_report[source_file] = {'error': f"Corresponding output file {output_file} does not exist."}



    # Print or process the report as needed
    for file, file_report in missing_links_report.items():
        print(f"File: {file}")
        if 'error' in file_report:
            print(f"Error: {file_report['error']}")
        else:
            print(f"Missing Links Count: {file_report['missing_links_count']}")
            # Uncomment below if you want to print the missing links
            # print(f"Missing Links: {file_report['missing_links']}")
        print()


#overview_source()
check_data_quality()
#compare_links()