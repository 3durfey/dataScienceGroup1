import os
import pickle

import streamlit as st
from dotenv import load_dotenv
from utils.dataclean_jagath import Clean
from utils.peter import CleanCityname
from utils.b2 import B2


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

columnNames = list(df_apartments.columns)
columnNames = ";".join(columnNames).split(";")


# ------------------------------
# Layout for Filters: State and Bedrooms
# ------------------------------

# Get unique states and max bedrooms
states = df_cleaned1['state'].unique()
max_bedrooms = int(df_cleaned1['bedrooms'].max())

# Display filters in one row
col1, col2 = st.columns(2)
with col1:
    selected_state = st.selectbox("Select a state:", states)
with col2:
    selected_bedrooms = st.selectbox("Select bedrooms (1 to max):", range(1, max_bedrooms + 1))

# Button to apply filters
if st.button("Show Filtered Apartments"):
    # Filter the data based on selections
    filtered_data = df_cleaned1[
        (df_cleaned1['state'] == selected_state) &
        (df_cleaned1['bedrooms'] == selected_bedrooms)
    ]

    # Display the filtered data as a single-row table
    st.write(f"Apartments in {selected_state} with {selected_bedrooms} bedrooms:")
    st.table(filtered_data)  # Only showing the first row for simplicity