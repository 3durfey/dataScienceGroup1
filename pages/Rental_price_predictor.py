# Import necessary libraries
import streamlit as st
import pandas as pd
from app import df_price_prediction as data
# App title
st.title("Rental Price Predictor")
# Step 1: Dropdown for square feet
# Split the single column into multiple columns based on commas
data[['square_feet', 'bedrooms', 'bathrooms', 'AmenityCount', 'predicted_price']] = data['square_feet,bedrooms,bathrooms,AmenityCount,predicted_price'].str.split(',', expand=True)

# Drop the original single column if it's no longer needed
data = data.drop(columns=['square_feet,bedrooms,bathrooms,AmenityCount,predicted_price'])

# Convert columns to appropriate data types (e.g., numeric)
data = data.apply(pd.to_numeric, errors='ignore')

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
