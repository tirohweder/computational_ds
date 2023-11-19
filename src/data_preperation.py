import pandas as pd

# Load the CSV
df = pd.read_csv("../data/output/cnn_2019_output.csv")

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'].str.extract('(\w+ \d{1,2}, \d{4})')[0])

# Save the modified DataFrame back to CSV (optional)
df.to_csv("data/output/cnn_2019_output2.csv", index=False)

# Display the DataFrame (optional)



print(df['Date'].isnull().sum())
print(df['Date'].apply(lambda x: not isinstance(x, pd.Timestamp)).sum())
non_datetime_rows = df[df['Date'].apply(lambda x: not isinstance(x, pd.Timestamp))]
print(non_datetime_rows)