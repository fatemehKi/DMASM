# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:54:41 2019

@author: gsaikia
"""
import numpy as np

#import dataset
from sklearn.datasets import load_boston
dataset = load_boston()

#extract independent and response variables
X = dataset.data
y = dataset.target

#scale the values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))

#split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#import model
from sklearn.linear_model import LinearRegression 
model = LinearRegression()

#train the model
model.fit(X_train,y_train)

#see performance score
model.score(X_test,y_test)

#get predicted values--- without looking at the y_test
y_pred = model.predict(X_test)

y_pred = sc_y.inverse_transform(y_pred.reshape(len(y_pred),1)).reshape(len(y_pred))
y_test = sc_y.inverse_transform(y_test.reshape(len(y_test),1)).reshape(len(y_test))



from sklearn.metrics import mean_squared_error #SSE/n
meansqerr = mean_squared_error(y_test,y_pred)
print(meansqerr)

###################################################################
#analysis of the individual co-efficients' values

import statsmodels.api as sm

X_modified = sm.add_constant(X_train)
lin_reg = sm.OLS(y_train,X_modified)
result = lin_reg.fit()
print(result.summary())

X_modified = np.delete(X_modified,3,axis=1)
lin_reg = sm.OLS(y_train,X_modified)
result = lin_reg.fit()
print(result.summary())


X_modified = np.delete(X_modified,6,axis=1)
lin_reg = sm.OLS(y_train,X_modified)
result = lin_reg.fit()
print(result.summary())












