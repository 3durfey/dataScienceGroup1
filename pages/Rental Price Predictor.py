# Import necessary libraries
import streamlit as st
import pandas as pd

# Load data into a DataFrame
url = "https://raw.githubusercontent.com/3durfey/dataScienceGroup1/refs/heads/main/data/predicted_price.csv"


def test_load_data(url):
    # Load the data
    df = pd.read_csv(url)

    # Check if the DataFrame is not empty
    assert not df.empty, "The DataFrame is empty. Data did not load correctly."

    # Check if the expected columns exist
    expected_columns = {'square_feet', 'bedrooms', 'bathrooms', 'predicted_price'}
    assert expected_columns.issubset(df.columns), f"Expected columns {expected_columns} are not all present in the DataFrame."

    # Check if the data types of specific columns are as expected
    assert pd.api.types.is_numeric_dtype(df['square_feet']), "'square_feet' column is not numeric."
    assert pd.api.types.is_numeric_dtype(df['bedrooms']), "'bedrooms' column is not numeric."
    assert pd.api.types.is_numeric_dtype(df['bathrooms']), "'bathrooms' column is not numeric."
    assert pd.api.types.is_numeric_dtype(df['predicted_price']), "'predicted_price' column is not numeric."


@st.cache_data
def load_data():
    # Read the CSV file from the provided URL
    df = pd.read_csv(url)
    return df

# Initialize data
data = load_data()

# App title
st.title("Rental Price Predictor")
st.write(
    """
    Welcome to the Rental Price Predictor! Use the dropdowns below to select the 
    apartment specifications, and the predicted rental price will be shown based on 
    your selections.
    """
)
# Step 1: Dropdown for square feet

st.write("### Step 1: Select Square Feet")
square_feet = st.selectbox("Select Square Feet", sorted(data['square_feet'].unique()))

# Step 2: Filter data based on square feet selection and populate bedrooms dropdown
st.write("### Step 2: Select Number of Bedrooms")
filtered_data_sqft = data[data['square_feet'] == square_feet]
bedrooms = st.selectbox("Select Bedrooms", sorted(filtered_data_sqft['bedrooms'].unique()))

# Step 3: Further filter data based on square feet and bedrooms selections and populate bathrooms dropdown
st.write("### Step 3: Select Number of Bathrooms")
filtered_data_bedrooms = filtered_data_sqft[filtered_data_sqft['bedrooms'] == bedrooms]
bathrooms = st.selectbox("Select Bathrooms", sorted(filtered_data_bedrooms['bathrooms'].unique()))

# Step 4: Final filter to get the predicted price based on all selections
st.write("### Predicted Rental Price")
final_filtered_data = filtered_data_bedrooms[filtered_data_bedrooms['bathrooms'] == bathrooms]

# Display the predicted price prominently
st.subheader("Predicted Price")
if not final_filtered_data.empty:
    predicted_price = final_filtered_data['predicted_price'].values[0]
    st.write(f"### ${predicted_price:,.2f}")
else:
    st.write("No predicted price available for the selected combination.")


st.write("## Additional Information")
st.write(
    """
    This app provides a predicted rental price based on historical data, helping 
    you make informed decisions about rental costs. Adjust the dropdown values to 
    see how the predicted price changes with different apartment specifications.
    """
)
