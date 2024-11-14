import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# data is loaded into df
import os
import pickle
 
import streamlit as st
from dotenv import load_dotenv
from utils.jagath import Clean, create_half_bathrooms
from utils.peter import CleanCityname
from utils.b2 import B2
from  dataclean_and_score.ScoreDistribution1_1  import ScoreDistribution 
 
 
# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'apartments_for_rent_classified_10K.csv'
 
 
# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()
# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])
 
 
# ------------------------------------------------------
#                        CACHING
# ------------------------------------------------------
@st.cache_data
def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_apartments = b2.get_df(REMOTE_DATA)
    return df_apartments
 
 
@st.cache_resource
def get_model():
    with open('./model.pickle', 'rb') as f:
        analyzer = pickle.load(f)
   
    return analyzer
 
# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
 
df_apartments = get_data()
# ------------------------------
# PART 1 : Filter Data
# ------------------------------
df_cleaned0 = Clean(df_apartments)
df_cleaned1 = CleanCityname(df_cleaned0)
df=df_cleaned1 

# Creating a price_per_sqft column  in df
df['price_per_sqft'] = df['price'] / df['square_feet'].replace(0, np.nan)
df = df.dropna(subset=['price_per_sqft'])



# Data preprocessing and model training
df = df.dropna(subset=['price', 'square_feet', 'bedrooms', 'bathrooms', 'cityname'])
X = df[['price_per_sqft', 'square_feet', 'bedrooms', 'bathrooms', 'cityname']]
X = pd.get_dummies(X, columns=['cityname'], drop_first=True)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_columns = X_train.columns

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


def predict_house_price(square_feet, bedrooms, bathrooms, cityname):
    avg_price_per_sqft = X_train['price_per_sqft'].mean()
    input_data = {
        'price_per_sqft': [avg_price_per_sqft],
        'square_feet': [square_feet],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
    }
    for city in model_columns[4:]:
        input_data[city] = 1 if f'cityname_{cityname}' == city else 0
    input_df = pd.DataFrame(input_data)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    predicted_price = model.predict(input_df)[0]
    return predicted_price


# Set up the title and input fields in Streamlit
st.title("House Price Prediction")

# User inputs
square_feet = st.number_input("Square Feet", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
cityname = st.selectbox("City Name", df['cityname'].unique())

# Button for prediction
if st.button("Predict House Price"):
    predicted_price = predict_house_price(square_feet, bedrooms, bathrooms, cityname)
    st.write(f"Predicted House Price: ${predicted_price:,.2f}")


# Calculate and display evaluation metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write("### Model Evaluation Metrics")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-Squared: {r2:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")


