import os
import pickle
from io import StringIO
import pandas as pd
import streamlit as st
import sys
sys.path.append('/opt/miniconda3/envs/dataScience/lib/python3.9/site-packages') 
from dotenv import load_dotenv
from utils.jagath import Clean
from utils.b2 import B2
from  dataclean_and_score.ScoreDistribution1_1  import ScoreDistribution 
from io import BytesIO
import io
import zipfile
from utils.jagath import PCA_PAIRWISE, FilteredData
# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
PICKLE_REMOTE_DATA = "10k_data.pickle"
PREDICTED_PRICES = 'df_price_prediction.pickle' 
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
def get_model(NAME):
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    model = b2.get_model(NAME)
    return model

@st.cache_resource
def get_pickle_model():
    with open('./random_forest_model_final.pkl', 'rb') as f:
        analyzer = pickle.load(f)
    
    return analyzer
sqft_model = get_pickle_model()
# ------------------------------------------------------
#                         APP BACKBLAZE
# ------------------------------------------------------

df_apartments = get_data(PICKLE_REMOTE_DATA)
df_price_prediction = get_data(PREDICTED_PRICES)

# ------------------------------------------------------
#                         APP LOCAL
# ------------------------------------------------------
def get_data_local(pickle_file_path):
    try:
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        if isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError("The loaded pickle file does not contain a Pandas DataFrame.")
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None
#df_apartments = get_data_local(PICKLE_REMOTE_DATA)
#df_price_prediction = get_data_local(PREDICTED_PRICES)

# Load the model from the ZIP file

# ------------------------------
# PART 1 : Filter Data
# ------------------------------

df_cleaned1 = Clean(df_apartments)
 
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


# ------------------------------
# Create st.session to hold apartments after user selects options
# ------------------------------
# Initialize session state for filtered data and recommended apartments
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = pd.DataFrame()
if "recommended_data" not in st.session_state:
    st.session_state.recommended_data = pd.DataFrame()
filtered_data = []
if "data" not in st.session_state:
    st.session_state.data = []
# Button to apply filters
if st.button("Show Filtered Apartments"):
    # Filter the data based on selections
    filtered_data = df_cleaned1[
    (df_cleaned1['state'] == selected_state) &
    #(df_cleaned1['bedrooms'] == selected_bedrooms)&
    (df_cleaned1['price']>=selected_price[0]) &
    (df_cleaned1['price']<= selected_price[1])]
    #&(df_cleaned1['bathrooms']== selected_bathrooms)
    chunk_size = 5
    # Split the DataFrame into chunks of 5 rows each
    df_chunks = [
        filtered_data.iloc[i:i + chunk_size]
        for i in range(0, len(filtered_data), chunk_size)
    ]
    # Store the list of DataFrames in session state
    st.session_state.data = df_chunks  # Overwrite with the new chunks
    
#-------------------------------------------------------------------------
#               SHOW APARTMENTS FROM FILTERED
#-------------------------------------------------------------------------
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
                st.markdown(f"**Time Listed:** {row['time']}")
            
            # Right column details
            with col2:
                st.markdown(f"**Amenities:** {row['amenities']}")
                st.markdown(f"**Pets Allowed:** {'Yes' if row['pets_allowed'] else 'No'}")
                st.markdown(f"**Cats Allowed:** {'Yes' if row['cats_allowed'] else 'No'}")
                st.markdown(f"**Dogs Allowed:** {'Yes' if row['dogs_allowed'] else 'No'}")
                st.markdown(f"**Has Photo:** {'Yes' if row['has_photo'] else 'No'}")
                st.markdown(f"**Latitude:** {row['latitude']}")
                st.markdown(f"**Longitude:** {row['longitude']}")
if "counter" not in st.session_state:
    st.session_state.counter = 0 

# Function to increment the counter
def increment_counter():
    if (st.session_state.counter < (len(st.session_state.data) - 1)):
        st.session_state.counter += 1  # Update session state variable

# Function to decrement the counter
def decrement_counter():
    if st.session_state.counter > 0:  # Prevent negative values
        st.session_state.counter -= 1

if len(st.session_state.data) > 0:
    # Layout for the buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            decrement_counter()
    with col2:
        if st.button("Next"):
            increment_counter()
            
# Check if the session state data has chunks and the counter is within range
if "data" in st.session_state and len(st.session_state.data) > 0:
    display_apartments(st.session_state.data[st.session_state.counter])
#-------------------------------------------------------------------------
#               GET SCORE BASED TOP APARTMENTS
#-------------------------------------------------------------------------


    # Assuming `df_filtered` is already available and contains all the columns provided
    # Ensure `df_filtered` is a pandas DataFrame
    #used current session df Jagath may change back to filtered_data
    df_filtered = pd.DataFrame(st.session_state.data[st.session_state.counter])
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



if st.button('More Recommended Apartments'):
    try:

        # Initialize FilteredData with selected filters
        FD = FilteredData(df_cleaned1, selected_state, selected_price, selected_bedrooms, selected_bathrooms)
        filtered_data = FD.filtered_data
        
        # Convert filtered data to DataFrame
        df_filtered = pd.DataFrame(filtered_data)

        # Ensure necessary columns exist
        if 'bathrooms' not in df_filtered.columns:
            st.error("The 'bathrooms' column is missing. Please check the data.")
            st.stop()
        if 'bedrooms' not in df_filtered.columns:
            st.error("The 'bedrooms' column is missing. Please check the data.")
            st.stop()

        # Calculate scores using ScoreDistribution
        bathroom_dis = ScoreDistribution(df_filtered['bathrooms'], selected_bathrooms, bathroom_rate)
        bedroom_dis = ScoreDistribution(df_filtered['bedrooms'], selected_bedrooms, bedroom_rate)
        np_score = bedroom_dis.apply_score() + bathroom_dis.apply_score()

        # Add score to the DataFrame and sort by score
        df_filtered['score'] = np_score
        df_filtered = df_filtered.sort_values(by='score', ascending=False)

        # Ensure df_filtered is not empty before proceeding
        if df_filtered.empty:
            st.warning("No recommended apartments match the selected criteria.")

        # Perform PCA and get similar apartments
        PPP = PCA_PAIRWISE(df_cleaned1)
        top5_indices = df_filtered.index[:5]
        top_similar = PPP.get_pairwise_dis(top5_index=top5_indices)

        # Display the recommended apartments
        display_apartments(df_cleaned1.loc[top_similar])
    
    except KeyError as e:
        st.error(f"A required column is missing: {e}")
    except ValueError as e:
        st.error(f"Value error encountered: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
