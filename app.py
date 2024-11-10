import os
import pickle

import streamlit as st
from dotenv import load_dotenv

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
    df_coffee = b2.get_df(REMOTE_DATA)
    print("test")  # Check the content before passing it to pd.read_csv()
    return df_coffee


@st.cache_resource
def get_model():
    with open('./model.pickle', 'rb') as f:
        analyzer = pickle.load(f)
    
    return analyzer

# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
st.write(
'''
# Review Sentiment Analysis
We pull data from our Backblaze storage bucket, and render it in Streamlit.
''')
df_apartments = get_data()

# ------------------------------
# PART 1 : Filter Data
# ------------------------------

st.write(
'''
**Your filtered data:**
''')

st.write(df_apartments.head(1))
# ------------------------------
# PART 2 : Plot
# ------------------------------

st.write(
'''
## Visualize
Compare this subset of reviews with the rest of the data.
'''
)


# ------------------------------
# PART 3 : Analyze Input Sentiment
# ------------------------------

st.write(
'''
## Custom Sentiment Check

Compare these results with the sentiment scores of your own input.
'''
)

