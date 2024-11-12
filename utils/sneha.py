from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score



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


def run_regression(df):
    features = ['AmenityCount','square_feet','bedrooms','bathrooms']
    target = 'price'


    X = df[features]
    y = df[target]

    encoder = OneHotEncoder(handle_unknown="ignore")
    # One-hot encode categorical features and scale numerical features
    preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['bedrooms','bathrooms','AmenityCount']),
    ('num', StandardScaler(), ['square_feet'])
    ])
 
 
        # Polynomial degree (adjust this as needed)
    poly = PolynomialFeatures(degree=2)
 
    # Create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', poly),
        ('regressor', LinearRegression())
    ])
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    print("Model trained successfully.")
 
    y_pred = pipeline.predict(X_test)
 
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
 
    return pipeline, y_test, y_pred, r2, mse

def generate_feature_data():
    # Generate square footage from 500 to 10000 with a step of 100
    sqft_values = np.arange(500, 10001, 100)
    data = []

    for sqft in sqft_values:
        # Set bedrooms based on square footage
        bedrooms = min(8, max(1, sqft // 1000))

        # Set bathrooms to be at least half of bedrooms, rounded up
        bathrooms = int(np.ceil(bedrooms / 2))

        # Randomize amenity count for variety
        amenity_count = np.random.randint(1, 10)

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


def predict_prices(feature_data, pipeline):
    # Predict prices using the trained pipeline
    predicted_prices = pipeline.predict(feature_data)

    # Create a new DataFrame to hold the features and predicted prices
    result_df = feature_data.copy()  # Start with the features DataFrame
    result_df['predicted_price'] = predicted_prices  # Add predictions as a new column

    return result_df