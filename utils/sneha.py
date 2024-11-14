from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st




def Clean(df):
    """
    Here, the DataFrame df column 'currency' is used as means for checking for 
    incorrect data entries
    """
    clean_df = df.copy()
    clean_df = clean_df[clean_df['currency']=='USD']
    clean_df.drop_duplicates(keep = 'first',inplace = True)
    if 'bathrooms' in clean_df.columns:
        clean_df['bathrooms'] = clean_df['bathrooms'].apply(lambda x: float(x))
    if 'bedrooms' in clean_df.columns:
        clean_df['bedrooms'] = clean_df['bedrooms'].apply(lambda x: float(x))
    if 'cats_allowed' not in clean_df.columns:
        clean_df['cats_allowed'] = clean_df['pets_allowed'].isin({'Cats,Dogs', 'Cats'})
    if 'dogs_allowed' not in clean_df.columns:
        clean_df['dogs_allowed'] = clean_df['pets_allowed'].isin({'Cats,Dogs', 'Dogs'})
 
    clean_df.reset_index(drop  = True,inplace = True)
    clean_df = clean_df
 
    print(f'Data cleaning is success, returning clean_df')
    return clean_df

# Function to train a regression model
def run_regression(df):
     
    """
    Trains a polynomial regression model to predict rental prices based on features.
    Returns the trained pipeline and evaluation metrics (R2 and MSE).
    """
     
    features = ['AmenityCount','square_feet','bedrooms','bathrooms'] # Define feature columns
    target = 'price' # Define target variable

    # Separate the features and target
    X = df[features]
    y = df[target]

    encoder = OneHotEncoder(handle_unknown="ignore")
    # One-hot encode categorical features and scale numerical features

    # Define preprocessor: One-hot encodes categorical and scales numerical features
    preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['bedrooms','bathrooms','AmenityCount']),
    ('num', StandardScaler(), ['square_feet'])
    ])
 
    # Polynomial transformation with degree 2
        # Polynomial degree (adjust this as needed)
    poly = PolynomialFeatures(degree=2)
 
    # Create a pipeline with preprocessing, polynomial transformation, and regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', poly),
        ('regressor', LinearRegression())
    ])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train) # Train the model pipeline
    print("Model trained successfully.")
    # Predict on test data
    y_pred = pipeline.predict(X_test)
    # Calculate Mean Squared Error (MSE) and R-Squared (R2) for model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
 
    return pipeline, y_test, y_pred, r2, mse # Return model pipeline and evaluation metrics


# Function to generate feature data for predictions
def generate_feature_data():
    """
    Generates synthetic feature data for testing predictions.
    Returns a DataFrame with square footage, bedrooms, bathrooms, and AmenityCount.
    """
    # Generate square footage from 500 to 10000 with a step of 100
    sqft_values = np.arange(500, 10001, 100)
    data = []

    # Loop through square footage values to generate bedrooms, bathrooms, and amenities
    for sqft in sqft_values:
        # Set bedrooms based on square footage
        bedrooms = min(8, max(1, sqft // 1000))

        # Set bathrooms to be at least half of bedrooms, rounded up
        bathrooms = int(np.ceil(bedrooms / 2))

        # Randomize amenity count for variety
        amenity_count = np.random.randint(1, 10)

        # Append generated data to list
        # Append the generated row to data
        data.append({
            'square_feet': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'AmenityCount': amenity_count
        })

    # Convert list of dictionaries to DataFrame
    feature_df = pd.DataFrame(data)
    return feature_df

# Function to predict prices using the trained pipeline
def predict_prices(feature_data, pipeline):
    """
    Predicts rental prices based on feature data and a trained model pipeline.
    Returns a DataFrame with features and predicted prices.
    """
    # Predict prices using the trained pipeline
    predicted_prices = pipeline.predict(feature_data)

    # Create a new DataFrame to hold the features and predicted prices
    result_df = feature_data.copy()  # Start with the features DataFrame
    result_df['predicted_price'] = predicted_prices  # Add predictions as a new column

    return result_df # Return DataFrame with predictions


# Helper function to count amenities
def count_valid_amenities(amenities):
    """
    Count the number of valid amenities in a comma-separated string.
    Returns the count of amenities for each entry.
    """
    if pd.isnull(amenities):
        return 0
    amenities_list = [item.strip() for item in amenities.split(',') if item.strip()]
    return len(amenities_list)


# Function to plot a box plot of price by amenity count
def plot_boxplot_price_by_amenity_count(data, max_price=3000):
    """
    Display a box plot of price by amenity count.
    Filters data by price and calculates amenity count for each row.
    
    Parameters:
    - data: pd.DataFrame containing 'amenities' and 'price' columns.
    - max_price: Maximum price value to filter the data by.
    """
    # Add AmenityCount column
    data['AmenityCount'] = data['amenities'].apply(count_valid_amenities)
    filtered_data = data[data['price'] < max_price]
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=filtered_data['AmenityCount'], y=filtered_data['price'])
    plt.xlabel('Amenity Count')
    plt.ylabel('Price')
    plt.title('Box Plot of Price by Amenity Count')
    
    # Display in Streamlit
    st.pyplot(plt)


# Function to plot a scatter plot of longitude vs latitude
def plot_scatter_longitude_latitude(data):
    """
    Display a scatter plot of longitude vs latitude.
    
    Parameters:
    - data: pd.DataFrame containing 'longitude' and 'latitude' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data['longitude'], data['latitude'], color='red', marker='o')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of Latitude vs Longitude')
    
    # Display in Streamlit
    st.pyplot(plt)

