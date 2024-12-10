import re
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

################################### Clean and create_half_bedrooms ################################

def Clean(df):
    """
    Here, the DataFrame df column 'currency' is used as means for checking for 
    incorrect data entries
    """
    clean_df = df.copy()
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

    clean_df = create_half_bathrooms(clean_df)
    clean_df = clean_df.fillna({'bathrooms':0, 'bedrooms':0, 'state':'ZZ'})

    print(f'Data cleaning is success, returning clean_df')
    return clean_df

def create_half_bathrooms(df):
    if 'half_bathrooms' not in df.columns:
        df['half_bathrooms'] = 0.0
        for r in df.index:
            try:
                if df.loc[r,'bathrooms']%1 != 0:
                    df.loc[r,'half_bathrooms'] = 1
                    df.loc[r,'bathrooms'] = float(int(df.loc[r,'bathrooms']))
            except:
                continue
    return df

############################################## PCA and Pairwise matrix ###############

COLUMNS_CONSIDERED = ['bathrooms', 'bedrooms', 'price', 'square_feet', 'state', 'latitude', 'longitude', 'cats_allowed', 'dogs_allowed']
class PCA_PAIRWISE:
    def __init__(self, clean_df):
        """_Takes Apartment Dataframe to calculate PCA with two features and later can be used to get pairwise distances

        Args:
            clean_df (_type_): a clean_df is preferred and Columns considered are ['bathrooms', 'bedrooms', 'price',
              'square_feet', 'state', 'latitude', 'longitude', 'cats_allowed', 'dogs_allowed'] for PCA calculation
        """
        self.clean_df = clean_df.copy()
        self.pcadf = self._get_df_numeric_columns()
        self.indices = self.pcadf.index
        self.new_df = self._perform_pca(self.pcadf)

    def _get_df_numeric_columns(self):
        """
        Take a DataFrame and return numeric columns of the dataframe

        Returns:
          Numerics columned DataFrame
        """
        pcadf = self.clean_df
        pcadf = pcadf.fillna({'bathrooms':0., 'bedrooms':0., 'state':'NAN'})
        all_columns = pcadf.columns
        for c in all_columns: 
            if c not in COLUMNS_CONSIDERED:
                try:
                    pcadf.drop(columns = [c], inplace = True)
                except:
                    print(f' Failed to drop {c} column ')
                    continue
        pcadf.dropna(inplace = True)
        pcadf = pd.get_dummies(pcadf,columns = ['state'],drop_first = True)
        return pcadf


    def _perform_pca(self,pcadf):
        """
        Takes numeric columns Dataframe to return PCA dataframe

        Args:
            pcadf (pandas.DataFrame):  only float(or int) columned Dataframe 

        Returns:
            PCA dataframe with two features
        """
        pca = PCA()
        new_np= pca.fit_transform(pcadf)
        #sigma_variance = pca.explained_variance_ratio_
        new_np = new_np[:,:2]   #only two new features considered
        new_df = pd.DataFrame(new_np, index = self.indices, columns = ['f1','f2'])
        return new_df

    def get_pairwise_dis(self,new_df = None, top5_index = None,return_paird = False, req_top = 5):
        """
        Calculates pairwise distances and return indices of the top req_top indices 

        Args:
            new_df (_type_, optional): Dataframe on which pairwise is performed. Defaults to PCA dataframe from object.
            top5_index (_type_, optional): Pairwise distances with respect to top5_index is provided. Defaults to None.
            return_paird (bool, optional): True when Pairwise distances is required. Defaults to False.
            req_top (int, optional): _description_. Defaults to 5.

        Returns:
            Index object of req_top indices
        """
        if new_df is None:
            new_df = self.new_df
        if top5_index is None:
            top5_index = new_df.index
        pair_d = pairwise_distances(new_df, new_df.loc[top5_index], n_jobs=-1)
        ## sum(axis = 1) or along columns
        temp = pd.DataFrame(pair_d, index = self.indices, columns = top5_index )
        temp['sum'] = temp.sum(axis = 1)
        temp = temp.sort_values(by = 'sum')
        ## sort them and get top
        if return_paird:
            return pair_d
        return temp.index[:req_top]



############################################ Simple Search based on column names #########

class Simple_Search():
    def __init__(self, df, column_name = ['title','address','cityname']):
        """
        Dataframe is required when initializing Simple_Search class, columns are optional

        Args:
            df (pd.DataFrame): df on which Simple Search needs to be performed
            column_name (list, optional):  Defaults to ['title','address','cityname'].
        """
        self.df = df.copy()
        self.indices = self.df.index
        self.column_name = column_name
        self._clean_column_text()
        self._cv_matrix()
 
    def _clean_column_text(self):
        """
        Removed special characters and combines all columns mentioned when initializing the object
        """
        if isinstance(self.column_name, list):
            self.df.loc[:,self.column_name] = self.df.loc[:,self.column_name].apply(lambda s: re.sub(r'[$,.:\/|%&*]','',s) if isinstance(s, str) else s)
        else:
            self.df.loc[ :,[self.column_name]] = self.df.loc[:,[self.column_name]].apply(lambda s: re.sub(r'[$,.:/\|$%&]', '', s) if isinstance(s, str) else s )
        try:
            self.df['all_text'] = self.df.apply( lambda row: f"{row['title']} "
                                        f"{row['address'] if pd.notna(row['address']) else ''} "
                                        f"{row['cityname'] if pd.notna(row['cityname']) else ''}",
                                        axis=1)
            self.column_name = 'all_text'
        except:
            self.column_name = 'title'
 
    def _cv_matrix(self):
        """
        fits a Count Vectorizer and stores within the class
        """
        corpus = []
        if self.column_name in self.df.columns:
            for i in self.df.index:
                corpus.append(self.df[self.column_name][i])
        self.cv = CountVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
        self.X =  self.cv.fit_transform(corpus)
 

    def get_top5_indices(self,text, top = 5, threshold = 0.1):
        """
        Return Indices of rows in df which are similar to text query. 

        Args:
            text (str): text query
            top (int, optional): Required number of indices. Defaults to 5.
            threshold (float, optional): cosine similairty threshold to consider returning indices. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        input_vector = self.cv.transform([text])
        scores = cosine_similarity(input_vector, self.X)
        if (scores>=threshold).sum():
            idx = scores.argsort()[0][-1:-(top+1):-1]
            return idx  
        return None