import os
import pickle
from io import StringIO
import pandas as pd

import streamlit as st
from dotenv import load_dotenv
from utils.jagath import Clean, PCA_PAIRWISE
from utils.peter import CleanCityname
from utils.b2 import B2
from  dataclean_and_score.ScoreDistribution1_1  import ScoreDistribution 
import datetime
 
# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'apartments_for_rent_classified_100K.csv'
# PREDICTED_PRICES = 'predicted_price.csv' 
 
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
def get_data(NAME):
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_apartments = b2.get_df(NAME)
    return df_apartments

 
@st.cache_resource
def get_model():
    with open('./model.pickle', 'rb') as f:
        analyzer = pickle.load(f)
   
    return analyzer
 
# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
 
df_apartments = get_data(REMOTE_DATA)
# df_price_prediction = get_data(PREDICTED_PRICES)

# ------------------------------
# PART 1 : Filter Data
# ------------------------------

df_cleaned0 = Clean(df_apartments)
df_cleaned1 = CleanCityname(df_cleaned0)
 
# ------------------------------
# Layout for Filters: State and Bedrooms , Price and Bathrooms
# ------------------------------
 
# Get unique states and max bedrooms
states = df_cleaned1['state'].unique()

max_bedrooms = int(df_cleaned1['bedrooms'].max())

max_bathrooms = int(df_cleaned1['bathrooms'].max())

min_price = int(df_cleaned1['price'].min())

max_price = int(df_cleaned1['price'].max())

# Display filters in one row
col1, col2 = st.columns(2)
with col1:
    selected_state = st.selectbox("Select a state:", sorted(states))
with col2:
    selected_bedrooms = st.selectbox(f"Select bedrooms (1 to {max_bedrooms}):", range(1, max_bedrooms + 1))

col3,col4 = st.columns(2)
with col3:
    selected_bathrooms = st.selectbox(f'Select bathrooms (1 to {max_bathrooms}):',range(1, max_bathrooms+1))
with col4:
    selected_price = st.slider(
    "Select price range:",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=50
)
col5,col6 = st.columns(2)
with col5:
    bedroom_rate = st.selectbox('Bedroom rating',range(10,0,-1))
with col6:
    bathroom_rate = st.selectbox('Bathroom rating',range(10,0,-1))

def display_apartments(data):

    for _, row in data.iterrows():
        st.markdown("---")  # Horizontal line to separate listings
        st.markdown(f"## 🏢 {row['title']}")  # Title of the apartment
        
        with st.container():
            col1, col2 = st.columns([2, 1])  # Adjusting column width ratios for better layout
            
            # Left column details
            with col1:
                st.markdown(f"**Description:** {row['body']}")
                st.markdown(f"**Square Feet:** {row['square_feet']} sqft")
                st.markdown(f"**Bedrooms:** {row['bedrooms']} 🛏️")
                st.markdown(f"**Bathrooms:** {row['bathrooms']} 🛁")
                st.markdown(f"**Half Bathrooms:** {row['half_bathrooms']}")
                st.markdown(f"**Price:** ${row['price']:,.2f}")
                st.markdown(f"**Price Type:** {row['price_type']} ({row['currency']})")
                st.markdown(f"**Fee:** ${row['fee']}")
                st.markdown(f"**Address:** {row['address']}, {row['cityname']}, {row['state']}")
                st.markdown(f"**Source:** {row['source']}")
                st.markdown(f"**Time Listed:** {datetime.datetime.fromtimestamp(row['time'])}")
            
            # Right column details
            with col2:
                st.markdown(f"**Amenities:** {row['amenities']}")
                st.markdown(f"**Pets Allowed:** {'Yes' if row['pets_allowed'] else 'No'}")
                st.markdown(f"**Cats Allowed:** {'Yes' if row['cats_allowed'] else 'No'}")
                st.markdown(f"**Dogs Allowed:** {'Yes' if row['dogs_allowed'] else 'No'}")
                st.markdown(f"**Has Photo:** {'Yes' if row['has_photo'] else 'No'}")
                st.markdown(f"**Latitude:** {row['latitude']}")
                st.markdown(f"**Longitude:** {row['longitude']}")

# Button to apply filters
if st.button("Show Filtered Apartments"):
    # Filter the data based on selections
    filtered_data = df_cleaned1[
        (df_cleaned1['state'] == selected_state) &
        (df_cleaned1['price']>=selected_price[0]) &
        (df_cleaned1['price']<= selected_price[1])
    ]

    # Display the filtered data as a single-row table
    st.write(f"Apartments in {selected_state} with {selected_bedrooms} bedrooms:")
    st.write("first")



#-------------------------------------------------------------------------
#               GET SCORE BASED TOP APARTMENTS
#-------------------------------------------------------------------------


    # Assuming `df_filtered` is already available and contains all the columns provided
    # Ensure `df_filtered` is a pandas DataFrame
    df_filtered = pd.DataFrame(filtered_data)
    ### Adding the scoring system/methods
    bathroom_dis = ScoreDistribution(df_filtered['bathrooms'], selected_bathrooms, bathroom_rate)
    bedroom_dis = ScoreDistribution(df_filtered['bedrooms'], selected_bedrooms, bedroom_rate)
    np_score = bedroom_dis.apply_score() + bathroom_dis.apply_score()
    df_filtered['score'] = np_score
    df_filtered = df_filtered.sort_values(by = 'score', ascending= False)
    #### df_cleaned1 is sorted based on the score
    if df_cleaned1.empty:
        st.error("No data available to display.")
        st.stop()  # Stop the script execution if no data
    # Function to display apartments in a custom card-like format
   
    # Pagination controls
    rows_per_page = 5  # Change this to control how many rows per page
    total_pages = -(-len(df_filtered) // rows_per_page)  # Ceiling division
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, value=1)

    # Get the data for the current page
    start_row = (current_page - 1) * rows_per_page
    end_row = start_row + rows_per_page
    df_paginated = df_filtered.iloc[start_row:end_row]
       ##need the top5 indices for PCA calculation
    # Display the paginated data in a custom format
    display_apartments(df_paginated)


#---------------------------------------------------------------------------------------------------------
#                            GET SIMILAR APARTMENT 
#---------------------------------------------------------------------------------------------------------



from utils.jagath import FilteredData

if st.button('More Recommended Apartments'):
    FD = FilteredData(df_cleaned1, selected_state,selected_price,selected_bedrooms, selected_bathrooms)
    filtered_data = FD.filtered_data
    df_filtered = pd.DataFrame(filtered_data)

    ##### repeated code block can be further reduced to a function in ScoreDistribution class###### need to work on it
    bathroom_dis = ScoreDistribution(df_filtered['bathrooms'], selected_bathrooms, bathroom_rate)
    bedroom_dis = ScoreDistribution(df_filtered['bedrooms'], selected_bedrooms, bedroom_rate)
    np_score = bedroom_dis.apply_score() + bathroom_dis.apply_score()
    df_filtered['score'] = np_score
    df_filtered = df_filtered.sort_values(by = 'score', ascending= False)
    ###################################


    PPP = PCA_PAIRWISE(df_cleaned1)
    top5_indices = df_filtered.index[:5]
    top_similar = PPP.get_pairwise_dis(top5_index= top5_indices)
    
    display_apartments(df_cleaned1.loc[top_similar])