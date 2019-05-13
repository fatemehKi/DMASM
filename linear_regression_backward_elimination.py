# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:54:49 2019

@author: mfatemeh
"""

from sklearn.datasets import load_boston
import numpy as np

dataset=load_boston()
#data is our x data and target is y
X=dataset.data
y=dataset.target

from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
sc_y=StandardScaler()

X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))

#splot into train and test set
#if we don't get the test size the default is 25% of the data for the test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0)

#import linear regression model 
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#train the model
model.fit(X_train, y_train)

#see the performance of the model #means 63 percent of the house price can be predicted
y_pred=model.score(X_test, y_test)

y_pred= model.predict(X_test)

y_pred=sc_y.inverse_transform(y_pred.reshape(len(y_pred),1)).reshape(len(y_pred))
y_test=sc_y.inverse_transform(y_test.reshape(len(y_test),1)).reshape(len(y_test))


from sklearn.metrics import mean_squared_error
meansqerr = mean_squared_error(y_test, y_pred)

#########################
#analysis of the individual co-efficients' value
import statsmodels.api as sm
X_modified = sm.add_constant(X_train)
lin_reg=sm.OLS(y_train, X_modified)
result=lin_reg.fit()
print(result.summary())

#how does t-square help.. tif the absolute value of t is below 2 then they are problematic
#if the p-value is higher it is not good and need to be removed

#we can see the x_3 is not healthy
X_modified=np.delete(X_modified,3,axis=1)
lin_reg=sm.OLS(y_train, X_modified)
result=lin_reg.fit()
print(result.summary())

#now we have a problem with x_6

X_modified=np.delete(X_modified,6,axis=1)
lin_reg=sm.OLS(y_train, X_modified)
result=lin_reg.fit()
print(result.summary())
