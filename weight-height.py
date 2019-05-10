#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('weight-height.csv')
dataset.drop('Gender',axis=1,inplace=True)

#correlation analysis
dataset.corrwith(dataset.Weight)

#extract independent and dependent variables
X = dataset.drop('Weight',axis=1) 
y = dataset.Weight

#scatter plot for X and y
plt.scatter(X,y)

#split the dataset into training-set and test-set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=0)

#import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#train model on the training-set
model.fit(X_train,y_train)

#r^2 score of the model on test-set
model.score(X_test,y_test)

#intercept (b0)
b0=model.intercept_

#slope (b1)
b1=model.coef_[0]

x = np.array([X.Height.min(),X.Height.max()])
y_pred = b0 + b1*x

#Visualisation of regression line and datapoints
#line plot
plt.plot(x,y_pred,c='red')
#scatter plot for X and y
plt.scatter(X,y,c='green')
plt.xlabel('Height in Inches')
plt.ylabel('Weight in Pounds')
plt.title('Simple Linear Regression')













