import matplotlib.pyplot as plt
import seaborn as sns

def sqft_plot(df):
    # Check if the 'price' column exists
    if 'price' in df:
        # Convert the 'price' column to numeric (if necessary) and drop NaN values
        #df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Set the style for seaborn
        sns.set_theme(style="whitegrid")


        # 1. Histogram of Square Feet
        plt.figure(figsize=(10, 6))
        sns.histplot(df['square_feet'], bins=30, kde=True, color='blue')
        plt.title('Distribution of Square Feet in Apartments')
        plt.xlabel('Square Feet')
        plt.ylabel('Frequency')
        plt.axvline(df['square_feet'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean Size')
        plt.axvline(df['square_feet'].median(), color='green', linestyle='dashed', linewidth=1, label='Median Size')
        plt.legend()
        plt.show()

        # 2. Box Plot of Rental Prices
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['price'], color='orange')
        plt.title('Box Plot of Rental Prices')
        plt.xlabel('Rental Price ($)')
        plt.show()
        


import pandas as pd

def clean_area_price(data):
    
    # Ensure 'square_feet' is numeric, replace invalid values with NaN
    data['square_feet'] = pd.to_numeric(data['square_feet'], errors='coerce')
    
    # Clean 'price' by removing non-numeric characters (e.g., "$", ",") and convert to numeric
    if 'price' in data.columns:
        data['price'] = data['price'].replace(r'[^\d.]', '', regex=True)
        data['price'] = pd.to_numeric(data['price'], errors='coerce')
    
    # Combine 'cityname' and 'state' into a 'location' column
    if 'cityname' in data.columns and 'state' in data.columns:
        data['location'] = data['cityname'] + ", " + data['state']
    else:
        data['location'] = None  # Handle missing columns gracefully
    
    # Perform one-hot encoding on the 'location' column
    if 'location' in data.columns:
        location_dummies = pd.get_dummies(data['location'], prefix='loc')
        data = pd.concat([data, location_dummies], axis=1)
        data.drop(columns=['location'], inplace=True, errors='ignore')
    
    return data

