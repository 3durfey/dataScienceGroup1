import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

class RentalPriceModel:
    def __init__(self, df):
        """
        Initialize the RentalPriceModel with a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        self.df = df
        self.cleaned_df = None
        self.pipeline = None

    def clean_data(self):
        """
        Cleans the DataFrame by filtering, removing duplicates, and handling missing columns.
        """
        try:
            clean_df = self.df.copy()

            # Filter rows based on currency
            if 'currency' in clean_df.columns:
                clean_df = clean_df[clean_df['currency'] == 'USD']
            else:
                raise KeyError("The column 'currency' is missing from the DataFrame.")

            # Remove duplicates
            clean_df.drop_duplicates(keep='first', inplace=True)

            # Handle bathroom and bedroom columns
            if 'bathrooms' in clean_df.columns:
                clean_df['bathrooms'] = pd.to_numeric(clean_df['bathrooms'], errors='coerce').fillna(0)
            if 'bedrooms' in clean_df.columns:
                clean_df['bedrooms'] = pd.to_numeric(clean_df['bedrooms'], errors='coerce').fillna(0)

            # Handle pet allowances
            if 'pets_allowed' in clean_df.columns:
                clean_df['cats_allowed'] = clean_df['pets_allowed'].str.contains('Cats', na=False)
                clean_df['dogs_allowed'] = clean_df['pets_allowed'].str.contains('Dogs', na=False)
            else:
                clean_df['cats_allowed'] = False
                clean_df['dogs_allowed'] = False

            clean_df.reset_index(drop=True, inplace=True)
            self.cleaned_df = clean_df
            print("Data cleaning is successful.")
            return self.cleaned_df
        except Exception as e:
            raise RuntimeError(f"An error occurred during data cleaning: {e}")

    def train_regression_model(self):
        """
        Trains a polynomial regression model to predict rental prices.
        Returns the trained pipeline and evaluation metrics (R2 and MSE).
        """
        try:
            if self.cleaned_df is None:
                raise ValueError("Data must be cleaned before training the model.")

            features = ['AmenityCount', 'square_feet', 'bedrooms', 'bathrooms']
            target = 'price'

            for column in features + [target]:
                if column not in self.cleaned_df.columns:
                    raise KeyError(f"The required column '{column}' is missing from the cleaned DataFrame.")

            # Separate features and target
            X = self.cleaned_df[features]
            y = self.cleaned_df[target]

            # Define preprocessor
            preprocessor = ColumnTransformer(transformers=[
                ('num', StandardScaler(), ['square_feet']),
                ('cat', OneHotEncoder(), ['bedrooms', 'bathrooms', 'AmenityCount'])
            ])

            # Polynomial transformation and pipeline
            poly = PolynomialFeatures(degree=2)
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('poly', poly),
                ('regressor', LinearRegression())
            ])

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.pipeline.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = self.pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("Model trained successfully.")
            return self.pipeline, y_test, y_pred, r2, mse

        except Exception as e:
            raise RuntimeError(f"An error occurred during model training: {e}")

    def predict_prices(self, feature_data):
        """
        Predicts rental prices based on feature data and the trained model pipeline.
        Returns a DataFrame with features and predicted prices.
        """
        try:
            if self.pipeline is None:
                raise ValueError("Model must be trained before making predictions.")

            if not isinstance(feature_data, pd.DataFrame):
                raise TypeError("Feature data must be a pandas DataFrame.")

            predicted_prices = self.pipeline.predict(feature_data)
            result_df = feature_data.copy()
            result_df['predicted_price'] = predicted_prices
            return result_df
        except Exception as e:
            raise RuntimeError(f"An error occurred during price prediction: {e}")

# Helper function to count amenities
def count_valid_amenities(amenities):
    """
    Count the number of valid amenities in a comma-separated string.
    Returns the count of amenities for each entry.
    """
    try:
        if pd.isnull(amenities):
            return 0
        amenities_list = [item.strip() for item in amenities.split(',') if item.strip()]
        return len(amenities_list)
    except Exception as e:
        raise ValueError(f"Error counting amenities: {e}")

# Function to plot a box plot of price by amenity count
def plot_boxplot_price_by_amenity_count(data, max_price=3000):
    """
    Display a box plot of price by amenity count.
    """
    try:
        if 'amenities' not in data.columns or 'price' not in data.columns:
            raise KeyError("Required columns 'amenities' and 'price' are missing from the data.")

        data['AmenityCount'] = data['amenities'].apply(count_valid_amenities)
        filtered_data = data[data['price'] < max_price]

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=filtered_data['AmenityCount'], y=filtered_data['price'])
        plt.xlabel('Amenity Count')
        plt.ylabel('Price')
        plt.title('Box Plot of Price by Amenity Count')

        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error creating box plot: {e}")

# Function to plot a scatter plot of longitude vs latitude
def plot_scatter_longitude_latitude(data):
    """
    Display a scatter plot of longitude vs latitude.
    """
    try:
        if 'longitude' not in data.columns or 'latitude' not in data.columns:
            raise KeyError("Required columns 'longitude' and 'latitude' are missing from the data.")

        plt.figure(figsize=(10, 6))
        plt.scatter(data['longitude'], data['latitude'], color='red', marker='o')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Scatter Plot of Latitude vs Longitude')

        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error creating scatter plot: {e}")
