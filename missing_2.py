# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:11:43 2019

@author: gsaikia
"""

import pandas as pd
import numpy as np

df = pd.read_csv('titanic_train_extra.csv')

df.isnull().sum()

dataset = df.dropna(thresh=7)

dataset.isnull().sum()

dataset.drop('Cabin',axis=1)

dataset.iloc[0,0] = np.nan
dataset.fillna(method='ffill',inplace=True)
dataset.isnull().sum()
dataset.fillna(method='bfill',inplace=True)


