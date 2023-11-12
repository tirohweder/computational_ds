import pandas as pd
import os


directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
print(directory)

def create_date():
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV and contains the required string in its name
        if filename.endswith('.csv') and ('cnn_' in filename or 'fox_' in filename or 'reuters_' in filename):

            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if the filename indicates it's a CNN file
            if 'cnn_' in filename:
                # Convert the 'Date' column to a datetime object for CNN and create a new 'timestamp' column
                df['timestamp'] = pd.to_datetime(df['Date'].str.extract('(\w+ \d{1,2}, \d{4})')[0])
                output_file = directory + os.sep + filename  # Adjust the naming as needed
            elif 'fox_' in filename:
                # Extract just the date information from the FOX date string
                df['timestamp'] = pd.to_datetime(df['Date'].str.extract('Published (\w+ \d{1,2}, \d{4})')[0],
                                                 format='Published %B %d, %Y', errors='coerce')
                output_file = directory + os.sep + filename

            elif 'reuters_' in filename:
                # Convert the 'Date' column to a datetime object for Reuters and create a new 'timestamp' column
                # Assuming the 'Date' column contains the ISO 8601 formatted timestamp string
                df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
                output_file = directory + os.sep + filename

            # Save the modified DataFrame back to CSV (if any changes are made)
            df.to_csv(output_file, index=False)

            # Output information about the 'Date' column
            print(f"Processing file: {filename}")
            print(f"Null Dates: {df['Date'].isnull().sum()}")
            print(f"Non-timestamp Dates: {df['Date'].apply(lambda x: not isinstance(x, pd.Timestamp)).sum()}")
            non_datetime_rows = df[df['Date'].apply(lambda x: not isinstance(x, pd.Timestamp))]
            print(non_datetime_rows)


create_date()

