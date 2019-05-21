# -*- coding: utf-8 -*-
"""
Created on Mon May 20 03:38:53 2019

@author: Kiaie
"""

import numpy as np
import pandas as pd
import seaborn as sns

###--- readig data set
ds1=pd.read_csv('Lego_set.csv')

###--- dropping unnecessary coloumns
ds2 = ds1.drop(['prod_desc', 'prod_long_desc', 'set_name', 'theme_name', 'prod_id', 'ages'], axis=1)


###--- handling missing values
ds2.isnull().sum() 
ds3=ds2.dropna(how='all')
ds3.fillna(method='bfill', inplace=True)

###---- correlation analysis
cor = ds3.corr()
corr = abs(cor)
sns.heatmap(corr,annot=True)


###--- encoding categorical data
dataset = pd.get_dummies(ds3,columns=['review_difficulty', 'country' ],drop_first=True)


###--- data and target sepertion
X=dataset.loc[:, (dataset.columns != 'list_price')].values
y=dataset.list_price.values

###---Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))


#Feature Elimination
from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import RFE
adj_R2 = []
feature_set = []
max_adj_R2_so_far = 0
n = len(X)
k = len(X[0])
for i in range(1,k+1):
    selector = RFE(LinearRegression(), i,verbose=1)
    selector = selector.fit(X, y)
    current_R2 = selector.score(X,y)
    current_adj_R2 = 1-(n-1)*(1-current_R2)/(n-i-1) 
    adj_R2.append(current_adj_R2)
    feature_set.append(selector.support_)
    if max_adj_R2_so_far < current_adj_R2:
        max_adj_R2_so_far = current_adj_R2
        selected_features = selector.support_
        selected_ranking= selector.ranking_
    print('End of iteration no. {}'.format(i))
    print('selector support is :', selector.support_)
    print('selected ranking is ;', selector.ranking_)

selected_ranking       
X_sub = X[:,selected_features]
#X_sub = X[:,:]

#split into train and test set.. the default is 25% of the data for the test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0)

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



from sklearn.metrics import mean_squared_error #SSE/n (MSE)
meansqerr = mean_squared_error(y_test,y_pred)
print(meansqerr)

#analysis of the individual co-efficients' values
import statsmodels.api as sm
X_modified = sm.add_constant(X_train)
lin_reg = sm.OLS(y_train,X_modified)
result = lin_reg.fit()
print(result.summary())



#4fold R2 value
scores = []
max_score = 0
from sklearn.model_selection import KFold
kf = KFold(n_splits=4,random_state=0,shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    current_model = LinearRegression()
    #train the model
    current_model.fit(X_train,y_train)
    #see performance score
    current_score = model.score(X_test,y_test)
    scores.append(current_score)
    if max_score < current_score:
        max_score = current_score
        best_model = current_model


best_model.intercept_
best_model.coef_
