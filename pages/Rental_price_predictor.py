# Import necessary libraries
import streamlit as st
import pandas as pd

# Load data into a DataFrame
url = "https://raw.githubusercontent.com/3durfey/dataScienceGroup1/refs/heads/main/data/predicted_price.csv"

@st.cache_data
def load_data():
    # Read the CSV file from the provided URL
    df = pd.read_csv(url)
    return df

# Initialize data
data = load_data()

# App title
st.title("Rental Price Predictor")

# Step 1: Dropdown for square feet
square_feet = st.selectbox("Select Square Feet", sorted(data['square_feet'].unique()))

# Step 2: Filter data based on square feet selection and populate bedrooms dropdown

filtered_data_sqft = data[data['square_feet'] == square_feet]
bedrooms = st.selectbox("Select Bedrooms", sorted(filtered_data_sqft['bedrooms'].unique()))

# Step 3: Further filter data based on square feet and bedrooms selections and populate bathrooms dropdown
filtered_data_bedrooms = filtered_data_sqft[filtered_data_sqft['bedrooms'] == bedrooms]
bathrooms = st.selectbox("Select Bathrooms", sorted(filtered_data_bedrooms['bathrooms'].unique()))

# Step 4: Final filter to get the predicted price based on all selections
final_filtered_data = filtered_data_bedrooms[filtered_data_bedrooms['bathrooms'] == bathrooms]

# Display the predicted price prominently
st.subheader("Predicted Price")
if not final_filtered_data.empty:
    predicted_price = final_filtered_data['predicted_price'].values[0]
    st.write(f"### ${predicted_price:,.2f}")
else:
    st.write("No predicted price available for the selected combination.")
