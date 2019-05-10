# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:39:22 2019

@author: mfatemeh
"""

import pandas as pd
dataset = pd.read_csv('weight-height.csv')

#remove unnecessary 
dataset.drop('Gender', axis=1, inplace=True)
dataset.corrwith(dataset.Weight)

#we don't need the coloumns below but we need a data set hterefore we delete individual as below.. we don't want one column.. we need one column of data frame
X= dataset.drop(['Weight'], axis=1)
y= dataset.Weight

# scatter plot for X and y
import matplotlib.pyplot as plt
plt.scatter(X,y)

# split the dataset into training-ser and test-set.. random_state is the seed  value to get the same value
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=.3, random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()

#train model on the training set
model.fit(X_train, y_train)

#r^2 score of the model on test set
model.score(X_test, y_test)

#we can give it multiple rows--- predicting the weight value given some height
model.predict([[50],[60],[70]])

#intercept (b0)
b0=model.intercept_

#slope (b1)
b1=model.coef_


#to plot we select the min x and max x
import numpy as np
x= np.array([X.Height.min(), X.Height.max()])
y_pred=b0+b1*x


#Visualization  of regression line and datapoint.. this visualization is possible due to being one variables.. for multiple varible we can't
plt.plot(x, y_pred, c='red')

#scatter plot for X and y
plt.scatter(X, y, c='green')
plt.xlabel('Height in Inches')
plt.ylabel('Weight in Pounds')
plt.title('Simple Linear Regression')
