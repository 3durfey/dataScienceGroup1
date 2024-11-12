#### dataclean functions #######
import pandas as pd
import numpy as np
import re


def CleanCityname(df):
# Drop rows where 'cityname' is null, contains numbers, or non-alphabetic characters
    cleaned = df[
        ~(df['cityname'].isnull() | 
        df['cityname'].str.contains(r'\d', na=False) | 
        df['cityname'].str.contains(r'[^a-zA-Z\s]', na=False))
    ]
    # Drop rows where any of the specified columns have null values
    df_apartments = df.dropna(subset=['state', 'latitude', 'longitude'])

    # Display the updated DataFrame to verify the changes
    print('city name cleaned')
  
    return cleaned




