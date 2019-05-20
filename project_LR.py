# -*- coding: utf-8 -*-
"""
Created on Mon May 20 03:38:53 2019

@author: Kian
"""

import numpy as np
import pandas as pd

dataset=pd.read_csv('Lego_set.csv')

###---checking missing number
dataset.isnull().sum() 
dataset=dataset.dropna(how='all')
dataset.fillna(method='bfill', inplace=True)

X=dataset.loc[:, (dataset.columns != 'list_price')]
X_1=X.drop(['ages'], axis=1)
y=dataset.list_price

###---
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
sc_y=StandardScaler()


X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))
