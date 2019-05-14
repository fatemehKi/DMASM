# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:49:02 2019

@author: mfatemeh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

print(dataset.DESCR)

X = dataset.data
y = dataset.target

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X= sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#train the model
model.fit(X_train,y_train)

#see performance score
model.score(X_test,y_test)

#get predicted values--- without looking at the y_test
y_pred = model.predict(X_test)

#performance measure for classification: accuracy = % correct predictios
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
