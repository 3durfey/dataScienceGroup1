#### dataclean functions #######
import pandas as pd
import numpy as np
import re

class DataClean():
    def __init__(self,path):
        assert(isinstance(path, str)),f'InvalidInput: path is not string'
        try:
            self.df = pd.read_csv('path', delimiter = ';', encoding_errors = 'ignore')
        except FileNotFoundError as fne:
            print(f'{fne}: file not found ')


    def Clean(self):
        """
        Here, the DataFrame df column 'currency' is used as means for checking for 
        incorrect data entries
        """
        clean_df = self.df
        clean_df = clean_df[clean_df['currency']=='USD']
        clean_df.drop_duplicates(keep = 'first',inplace = True)
        if 'bathrooms' in clean_df.columns:
            clean_df['bathrooms'] = clean_df['bathrooms'].apply(lambda x: float(x))
        if 'bedrooms' in clean_df.columns:
            clean_df['bedrooms'] = clean_df['bedrooms'].apply(lambda x: float(x))
        if 'cats_allowed' not in clean_df.columns:
            clean_df['cats_allowed'] = clean_df['pets_allowed'].isin({'Cats,Dogs', 'Cats'})
        if 'dogs_allowed' not in clean_df.columns:
            clean_df['dogs_allowed'] = clean_df['pets_allowed'].isin({'Cats,Dogs', 'Dogs'})

        ### adding any additional cleaning steps from other columns
        ### Can also add required columns
        ### can drop unnecessary columns
        clean_df.reset_index(inplace = True, drop  = True)
        self.clean_df = clean_df

        print(f'Data cleaning is success, returning clean_df')
        return self.clean_df
    
    
    



    