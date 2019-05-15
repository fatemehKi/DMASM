# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:53:40 2019

@author: mfatemeh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
dataset=load_iris()

print(dataset.DESCR)

#extract the independent and dependent variable
X = dataset.data
y = dataset.target

#findingg correlation among all (including target)
import seaborn as sns
df = pd.DataFrame(np.concatenate((X,y.reshape(len(y),1)),axis=1))
corr = df.corr()
sns.heatmap(corr,annot=True)

#we are not always scaling.. if we have the diference more than two digits we need to scale
 
#sploiting the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#train the model
model.fit(X_train,y_train)#training

#predict testcases
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)####********** to get the probability of eacjh

#see performance score on the test set
from sklearn.metrics import recall_score, precision_score, classification_report, confusion_matrix, accuracy_score

accuracy_score(y_test,y_pred) #how many prediction matches with the values (13+11+9 in confusion metric)
confusion_matrix(y_test, y_pred)
classification_report(y_test,y_pred)
print(classification_report(y_test,y_pred))


#like backward elimination we can use a method to automate it RFE=recursive feature elimination class) 
#RFE needs the model (Logistic reg), the number of feature that I need to drop, and step is the number of feature that
#is going to be eliminated in each step.. it doesn't check P-value it checks coefficient
#selector = RFE(estimator, i, step=1 )
from sklearn.feature_selection import RFE
estimator= LogisticRegression()

feature_set =[]
accuracies=[]
max_accuracy_so_far=0
for i in range(1,len(X[0])+1):
    selector = RFE(estimator, i, verbose=1 )
    selector =selector.fit(X,y)
    current_accuracy=selector.score(X,y)
    accuracies.append(current_accuracy)
    feature_set.append(selector.support_)
    if max_accuracy_so_far<current_accuracy:
        max_accuracy_so_far=current_accuracy
        slected_features=selector.support_        
    print('End of the iteration no.{}'.format(i))

