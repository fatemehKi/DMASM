# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:06:03 2019

@author: mfatemeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.contingency import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


dataset=pd.read_csv('data_banknote_authentication.txt', header=None, names=['variance', 'skewness', 'entropy', 'class']) 


#checking for missing
dataset.isnull().sum()


#description statistics
dataset.describe()





X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0)

training_score=[]
test_score=[]
for i in range(1,21):
    dt=DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
    dt.fit(X_train, y_train)
    y_pred_dt=dt.predict(X_test)
    training_score.append(dt.score(X_train, y_train))
    test_score.append(dt.score(X_test, y_test))
    
plt.plot(range(1,21), training_score, c='red')
plt.plot(range(1,21), test_score, c='green')
plt.xlabel('max_depth')




