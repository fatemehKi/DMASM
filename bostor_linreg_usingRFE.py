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


#split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sub,y,random_state=0)

#create model

model = LinearRegression()

#train the model
model.fit(X_train,y_train)

#see performance score
model.score(X_test,y_test)


#get predicted values
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









