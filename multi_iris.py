import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
from sklearn.datasets import load_iris
dataset = load_iris()

#extract independent and target variables
X = dataset.data
y = dataset.target

import seaborn as sns

df = pd.DataFrame(np.concatenate((X,y.reshape(len(y),1)),axis=1))

corr = df.corr()

sns.heatmap(corr,annot=True)




#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#import learning model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#train the model
model.fit(X_train,y_train) #training
model.feature_importance_

#predict testcases
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)

#performance measures on the test set
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score 

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))


from sklearn.feature_selection import RFE
accuracies = []
feature_set = []
max_accuracy_so_far = 0
for i in range(1,len(X[0])+1):
    selector = RFE(LogisticRegression(), i,verbose=1)
    selector = selector.fit(X, y)
    current_accuracy = selector.score(X,y)
    accuracies.append(current_accuracy)
    feature_set.append(selector.support_)
    if max_accuracy_so_far < current_accuracy:
        max_accuracy_so_far = current_accuracy
        selected_features = selector.support_
    print('End of iteration no. {}'.format(i))
        



















        

























































