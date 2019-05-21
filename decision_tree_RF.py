# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:41:13 2019

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

#correlation analysus 1.. this method we don't need to encode because it will automatically remove those
dataset.corr()


#correlation analysus 2
cor =abs(dataset.corr(method='spearman'))
sns.heatmap(cor,annot=True)


#if we had categorical we could use chi2

##########################WRONG######################
from scipy.stats.contingency import chi2_contingency
for i in range(len(X.iloc[0])):
    contigency_table = pd.crosstab(y, X.iloc[:,i])
    chi_2, p_val, dof, expected_val= chi2_contingency(contigency_table)
# if p_value is less than alpha (ass)

    alpha=0.05
    if p_val< alpha:
        print('The varianles are correlated at significant level', alpha)
    else:
        print('The variables are independent at signifiacant level', alpha)

#result showing the variables are not related to the target a\t all. pprobably together they are showing the 
##########################WRONG######################

X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0)

#create your model
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()

#train the model
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

#test the model
y_pred_dt =dt.predict(X_test)
y_pred_rf =rf.predict(X_test)

#see the model accuracy
dt.score(X_train, y_train)
rf.score(X_train, y_train)


dt.score(X_test, y_test)
rf.score(X_test, y_test)

### we don't 