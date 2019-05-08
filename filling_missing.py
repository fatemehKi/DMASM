# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:15:30 2019

@author: mfatemeh
"""

import pandas as pd
import numpy as np

df=pd.read_csv('titanic_train_extra_missing.csv')

#finding the number of missing
df.isnull().sum()

#to delete the row with missing in all columns.. the default is any
dataset=df.dropna(how='all')

dataset2=df.dropna(thresh=7)

#drop a column that has 
dataset.dropna(thresh=300, axis=1,inplace=True)
#in future because the dropna axis does not exist in future we just need to drop it using axis
dataset.drop('Cabin', axis=1)

dataset.fillna(method='ffill', inplace=True)

#to create a missinf value in the first row
dataset.iloc[0,0]=np.nan
#now the forward fillinf does not help
dataset.fillna(method='bfill', inplace=True)
