import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

print(dataset.DESCR)

X = dataset.data
y = dataset.target 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train) #training

model.score(X_test,y_test)

#testing
y_pred = model.predict(X_test) 

#performance measure for classification: accuracy = % correct predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

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

#Plotting ROC
y_pred_probs = model.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(y_test,y_pred_probs)

plt.plot(fpr,tpr)
plt.xlabel("FPR")
plt.ylabel('TPR')

plt.title('ROC with AUC score: {}'.format(roc_auc_score(y_test,y_pred_probs)))
plt.show()



from sklearn.preprocessing import binarize
y_pred_modified = binarize(y_pred_probs.reshape(len(y_test),1),threshold=0.7)
confusion_matrix(y_test,y_pred_modified)


















