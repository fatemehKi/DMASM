# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:31:53 2019

@author: mfatemeh
"""

import pandas as pd
df=pd.read_csv('Data.csv')

#fillling the missing value with the 
df.fillna({
                'Age':df.Age.median(),#using the most common values; however, it does have a conflict with the state because that city may not be in that province                
                'Salary' :df.Salary.mean()
                }, inplace=True)


#Change the country using dummy and the purchased using the map
dataset=pd.get_dummies(df, columns=['Country'], drop_first=True) # if the first drop was there it is hard to figure at which dummy is related to each column and the 
#dataset=pd.get_dummies(df, columns=['Country'])


#to change the purchased column saying No means 0 and yes means 1
dataset.Purchased=dataset.Purchased.map({'No':0, 'Yes':1})


from sklearn.preprocessing import StandardScaler #importing the class
#creating an instance from the class 
sc=StandardScaler()
#the scaled data is not the dataframe it is a series
dataset_scaled=sc.fit_transform(dataset)
#in order to inverse the scaling we do the below 
sc.inverse_transform(dataset_scaled)


#another way of scaling is min-max scalinf which is not suggested because it is not a good scaling
from sklearn.preprocessing import MinMaxScaler
MMS=MinMaxScaler()
dataset_scaled=MMS.fit_transform(dataset)
