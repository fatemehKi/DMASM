# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:59:59 2019

@author: gsaikia
"""

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

training_scores = []
test_scores = []
for i in range(1,21):
    #create your model
    dt = DecisionTreeClassifier(criterion='entropy',max_depth=i,random_state=0)
    #train the model
    dt.fit(X_train,y_train)
    #test the model
    y_pred_dt = dt.predict(X_test)
    #see model accuracy
    training_scores.append(dt.score(X_train,y_train))
    test_scores.append(dt.score(X_test,y_test))


plt.plot(range(1,21),training_scores,c='red')    
plt.plot(range(1,21),test_scores,c='green')
plt.xlabel('max_depth')
plt.ylabel('Model_score') 
plt.title('Model Score vs Maximum Depth Threshold in a Decision Tree')   
