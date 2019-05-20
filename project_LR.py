# -*- coding: utf-8 -*-
"""
Created on Mon May 20 03:38:53 2019

@author: Kian
"""

import numpy as np
import pandas as pd

dataset=pd.read_csv('Lego_set.csv')

###---checking missing number
dataset.isnull().sum() 
dataset=dataset.dropna(how='all')
dataset.fillna(method='bfill', inplace=True)
ds=dataset.drop(['ages'], axis=1)


dataset_2 = pd.get_dummies(ds,columns=['review_difficulty', 'country' ],drop_first=True)
dataset_3 = dataset_2.drop(['prod_desc', 'prod_long_desc', 'set_name', 'theme_name', 'prod_id'], axis=1)

X=dataset_3.loc[:, (dataset_3.columns != 'list_price')].values
y=dataset_3.list_price.values

###---
###---
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
sc_y=StandardScaler()

X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))


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
    print('End of iteration no. {}'.format(i))
        
X_sub = X[:,selected_features]



#splot into train and test set
#if we don't get the test size the default is 25% of the data for the test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0)


#import model
from sklearn.linear_model import LinearRegression 
model = LinearRegression()

#train the model
model.fit(X_train,y_train)


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
