
#pip install ucimlrepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
apartment_for_rent_classified = fetch_ucirepo(id=555) 
  
# data (as pandas dataframes) 

X = apartment_for_rent_classified.data.features 

X.isnull()
print(X.isnull().sum())


