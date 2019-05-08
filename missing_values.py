# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:06:23 2019

@author: gsaikia
"""

import pandas as pd

ufo = pd.read_csv('ufo.csv')
dataset = pd.read_csv('ufo.csv')

#descriptions
descript = ufo.describe()


#Check missing items in the dataset
ufo.isnull().sum() 

ufo['Colors Reported'].isnull().sum()
ufo['Colors Reported'].value_counts()
ufo['Colors Reported'].fillna('UNKNOWN',inplace=True)

ufo['Shape Reported'].isnull().sum()
ufo['Shape Reported'].value_counts()
ufo['Shape Reported'].fillna('UNKNOWN',inplace=True)

ufo.dropna(inplace=True)

dataset.describe()
dataset.isnull().sum()

dataset.fillna({
                   'Colors Reported':'Unknown',
                   'Shape Reported':'Unknown',
                },inplace=True)



dataset.City = dataset.groupby('State').City.apply(lambda x: x.fillna(x.mode().iloc[0]))


