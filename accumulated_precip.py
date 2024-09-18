import pandas as pd

# Assuming your dataset is in a pandas DataFrame with a 'date' column in YYYY-MM-DD format and a 'precipitation' column
# Replace with the actual column names if different.

# Example of data format
# data = pd.DataFrame({
#     'date': ['2000-01-01', '2000-02-01', '2000-03-01', ...],
#     'precipitation': [...]
# })

def process_precipitation(data):
    # Convert 'date' column to datetime format
    data['time'] = pd.to_datetime(data['time'])

    # Extract year and month from the 'date' column
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month

    # Set summer months (June, July, August, September) equal to 0
    data.loc[data['month'].isin([6, 7, 8, 9]), 'mean_swe'] = 0

    # Dictionary to hold the number of days in each month
    days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    # Multiply each value by the number of days in that month
    data['monthly_total_swe'] = data['mean_swe'] * data['month'].map(days_in_month)

    # Initialize the accumulated precipitation column
    data['accumulated_swe'] = 0

    # Calculate accumulated precipitation for each winter period
    for year in data['year'].unique():
        winter_data = data[(data['year'] == year) & (data['month'] >= 9) | ((data['year'] == year + 1) & (data['month'] <= 5))]
        accumulated_precip = 0
        for index, row in winter_data.iterrows():
            accumulated_precip += row['monthly_total_swe']
            data.loc[index, 'accumulated_swe'] = accumulated_precip

    return data

# Example usage
# Load data from a CSV file
data = pd.read_csv('Data/CMIP6_ssp585_swe/CMIP6_ssp585_mean_swe_dataset.csv')

# Process the data
processed_data = process_precipitation(data)

# Save the processed data to a CSV file
processed_data.to_csv('Data/CMIP6_ssp585_swe/CMIP6_ssp585_processed_swe.csv', index=False)

print(processed_data)
