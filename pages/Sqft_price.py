import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from app import df_cleaned1 as df
from app import sqft_model
# Creating a price_per_sqft column  in df
df['price_per_sqft'] = df['price'] / df['square_feet']
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
model = sqft_model


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
st.title("Apartment Price Prediction")
st.markdown("This page will provide the user with the 'Price' that is predicted based on the area covered(in sq.ft) and number of bedrooms and bathrooms, cityname ")

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
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)

# st.write("### Model Evaluation Metrics")
# st.write(f"Mean Squared Error: {mse:.2f}")
# st.write(f"R-Squared: {r2:.2f}")
# st.write(f"Mean Absolute Error: {mae:.2f}")


