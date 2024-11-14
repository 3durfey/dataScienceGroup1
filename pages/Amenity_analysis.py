import streamlit as st
import pandas as pd
from utils.sneha import plot_boxplot_price_by_amenity_count, plot_scatter_longitude_latitude
from app import df_cleaned1 as dataFrame

# Streamlit app layout
st.title("Apartment for Rent Data Visualization")

#sidebar setting 
st.sidebar.header("Plot Settings")
max_price = st.sidebar.slider("Max Price for Box Plot", min_value=500, max_value=5000, value=3000, step=100)

# Box Plot of Price by Amenity Count
st.subheader("Box Plot of Price by Amenity Count")
plot_boxplot_price_by_amenity_count(dataFrame, max_price)

# Box Plot Section
st.write("## Price Distribution by Amenity Count")
st.write(
    """
    This box plot shows the relationship between the number of amenities in an apartment and its rental price.
    Use the slider in the sidebar to adjust the maximum price shown, which allows you to focus on properties within a certain price range.
    """
)


# Scatter Plot of Longitude vs Latitude
st.subheader("Scatter Plot of Latitude vs Longitude")
plot_scatter_longitude_latitude(dataFrame)

st.write("## Geographical Distribution of Apartments")
st.write(
    """
    This scatter plot shows the distribution of apartments based on their geographical coordinates (longitude and latitude).
    Each red dot represents an apartment location. This plot can help identify clusters or distribution patterns in various regions.
    """
)
st.write("# Additional Information")
st.write(
    """
    The dataset provides insights into the apartment rental market, including variables like square footage, 
    number of bedrooms, bathrooms, and amenity count, along with geographic coordinates.
    You can explore relationships between these features using the visualizations above.
    """
)