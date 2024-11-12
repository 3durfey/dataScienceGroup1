# app.py
import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from utils.sneha import plot_boxplot_price_by_amenity_count, plot_scatter_longitude_latitude

# Fetch dataset
apartment_for_rent_classified = fetch_ucirepo(id=555)
X = apartment_for_rent_classified.data.features

# Streamlit app layout
st.title("Apartment for Rent Data Visualization")

st.sidebar.header("Plot Settings")
max_price = st.sidebar.slider("Max Price for Box Plot", min_value=500, max_value=5000, value=3000, step=100)

# Box Plot of Price by Amenity Count
st.subheader("Box Plot of Price by Amenity Count")
plot_boxplot_price_by_amenity_count(X, max_price)

# Scatter Plot of Longitude vs Latitude
st.subheader("Scatter Plot of Latitude vs Longitude")
plot_scatter_longitude_latitude(X)
