# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:07:09 2019

@author: mfatemeh
"""

import pandas as pd

# if there is no header we need to mention it 
ufo= pd.read_csv('ufo.csv')

#descriptions
descript=ufo.describe()
#count showing the number of non-missing values, unique number of distinct in each columns, top showing the "mode" of each columns
#freq is showing the number of 

#checking the number of missing
ufo.isnull().sum()

ufo['Colors Reported'].value_counts()
#the missing values are filled bu 
ufo['Colors Reported'].fillna('UNKNOWN', inplace=True)




ufo['Shape Reported'].isnull().sum()
ufo['Shape Reported'].value_counts()
#the missing values are filled bu 
ufo['Shape Reported'].fillna('UNKNOWN', inplace=True)

#droping rows with missing value in the city.. because we already replaced the other missings with UNKNOWN
ufo.dropna(inplace=True)



# to fill the columns at once we can use dictionary and for the city we use most frequent than the remove
ufo= pd.read_csv('ufo.csv')

#we have missing in city, shape Reported and colour reported
ufo.fillna({
                 'Colors Reported': 'Unknown',
                 'Shape Reported': 'Unknown',
                 'City':ufo.City.mode().iloc[0]  #using the most common values; however, it does have a conflict with the state because that city may not be in that province                
              }, inplace=True)

#another way of filling city with considering the state that
#first it group by state and then apply the function inside the group
ufo.City=ufo.groupby('State').City.apply(lambda x: x.fillna(x.mode().iloc[0]))


