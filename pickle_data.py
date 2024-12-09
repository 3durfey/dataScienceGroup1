import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import chardet




# Path to your CSV file
csv_file_path = 'apartments_for_rent_classified_10K.csv'

# Specify the correct delimiter
#df = pd.read_csv("apartments_for_rent_classified_10K.csv", sep=';')
# Check the columns to ensure they are split correctly
#print(df.columns)
df = pd.read_csv(csv_file_path, encoding='Windows-1252', sep=';',on_bad_lines='skip')

# Save the DataFrame as a pickle file
pickle_file_path = '10k_data.pickle'
df.to_pickle(pickle_file_path)
