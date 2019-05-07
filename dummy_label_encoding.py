# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:28:03 2019

@author: gsaikia
"""

import pandas as pd
df = pd.read_csv('titanic_train.csv')
df.head() #see first 5 rows in df

#Dummy variable encoding of one variable
dummies = pd.get_dummies(df.Embarked)
dummies.drop('C',axis=1,inplace=True)
dataset = df.drop('Embarked',axis=1)
dataset = pd.concat([dataset,dummies],axis=1)

#Dummy variable encoding of multiple variables
dataset_2 = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)

#Look up frequncies of unique values in a variable
df.Embarked.value_counts()

#Label Encoding used for ordinal variables
dataset_3 = df.copy(deep=True)
dataset_3.Embarked = dataset_3.Embarked.map({'C':1,'Q':2,'S':3})
dataset_3.Sex = dataset_3.Sex.map({'male':0,'female':1})







