# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:30:29 2019

@author: gsaikia
"""
'''
1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous) 
3. kurtosis of Wavelet Transformed image (continuous) 
4. entropy of image (continuous)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.contingency import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


#import dataset
df = pd.read_csv('data_banknote_authentication.txt',header=None,
                 names=['variance','skewness','kurtosis','entropy','class'])


#check for missing values
df.isnull().sum()

#descriptive stats
descript = df.describe()


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#correlation study
cor = abs(df.corr(method='spearman'))

sns.heatmap(cor,annot=True)

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#create your model
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

#train the model
dt.fit(X_train,y_train)
rf.fit(X_train,y_train)

#test the model
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)


#see model accuracy
dt.score(X_train,y_train)
rf.score(X_train,y_train)

dt.score(X_test,y_test)
rf.score(X_test,y_test)
