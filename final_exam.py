# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:58:58 2019

@author: fatemeh Kiaie
@description: Data Mining Final Exam
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('exam_dataset.csv')

####----- preprocessing
dataset.isnull().sum()
dataset.fillna(method='bfill', inplace=True)

###--- data and target sepertion
X=dataset.loc[:, (dataset.columns != 'Occupancy')].values
y=dataset.Occupancy.values

###---scaling the independent
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X = sc_X.fit_transform(X)

###---splitting to the test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0)

###--- training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)

model.fit(X_train,y_train) 

model.score(X_test,y_test)

#--- testing
y_pred = model.predict(X_test) 

#--- performance measure for classification: accuracy = % correct predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#--- confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import recall_score, precision_score
recall_score(y_test,y_pred) #sensitivity
precision_score(y_test,y_pred)

TN = cm[0,0]
FP = cm[0,1] 
FN = cm[1,0]
TP = cm[1,1]


FPR = FP/(FP+TN)
specificity = 1 - FPR

#--- Plotting ROC
y_pred_probs = model.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(y_test,y_pred_probs)

plt.plot(fpr,tpr)
plt.xlabel("FPR")
plt.ylabel('TPR')

plt.title('ROC with AUC score: {}'.format(roc_auc_score(y_test,y_pred_probs)))
plt.show()

###--- improving the model using binarize
from sklearn.preprocessing import binarize
y_pred_modified = binarize(y_pred_probs.reshape(len(y_test),1),threshold=0.7)
confusion_matrix(y_test,y_pred_modified)



#--- 5fold model scoring
scores = []
max_score = 0
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=0,shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    current_model = LogisticRegression(random_state=0)
    #train the model
    current_model.fit(X_train,y_train)
    #see performance score
    current_score = model.score(X_test,y_test)
    scores.append(current_score)
    if max_score < current_score:
        max_score = current_score
        best_model = current_model

