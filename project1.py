# -*- coding: utf-8 -*-
"""
Created on Sat May 18 00:48:44 2019

@author: Kian
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#----------------data loadig
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\bending1")
df1_b1=pd.read_csv('dataset1.csv')
df2_b1=pd.read_csv('dataset2.csv')
df3_b1=pd.read_csv('dataset3.csv')
df4_b1=pd.read_csv('dataset4.csv')
df5_b1=pd.read_csv('dataset5.csv')
df6_b1=pd.read_csv('dataset6.csv')
df7_b1=pd.read_csv('dataset7.csv')

df_b1=df1_b1.append(df2_b1.append(df3_b1.append(df4_b1.append(df5_b1.append(df6_b1.append(df7_b1, ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
df_b1['detected_activity']='bending_1'

###bending_2
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\bending2")
df1_b2=pd.read_csv('dataset1.csv')
df2_b2=pd.read_csv('dataset2.csv')
df3_b2=pd.read_csv('dataset3.csv')
df4_b2=pd.read_csv('dataset4.csv')
df5_b2=pd.read_csv('dataset5.csv')
df6_b2=pd.read_csv('dataset6.csv')

df_b2=df1_b2.append(df2_b2.append(df3_b2.append(df4_b2.append(df5_b2.append(df6_b2, ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
df_b2['detected_activity']='bending_2'

###cycling
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\cycling")
df1_c=pd.read_csv('dataset1.csv')
df2_c=pd.read_csv('dataset2.csv')
df3_c=pd.read_csv('dataset3.csv')
df4_c=pd.read_csv('dataset4.csv')
df5_c=pd.read_csv('dataset5.csv')
df6_c=pd.read_csv('dataset6.csv')
df7_c=pd.read_csv('dataset7.csv')
df8_c=pd.read_csv('dataset8.csv')
df9_c=pd.read_csv('dataset9.csv')
df10_c=pd.read_csv('dataset10.csv')
df11_c=pd.read_csv('dataset11.csv')
df12_c=pd.read_csv('dataset12.csv')
df13_c=pd.read_csv('dataset13.csv')
df14_c=pd.read_csv('dataset14.csv')
df15_c=pd.read_csv('dataset15.csv')

df_c=df1_c.append(df2_c.append(df3_c.append(df4_c.append(df5_c.append(df6_c.append(df7_c.append(df8_c.append(df9_c.append(df10_c.append(df11_c.append(df12_c.append(df13_c.append(df14_c.append(df15_c, ignore_index=True), ignore_index=True), ignore_index=True),ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True ), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
df_c['detected_activity']='cycling'


###lying
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\lying")
df1_l=pd.read_csv('dataset1.csv')
df2_l=pd.read_csv('dataset2.csv')
df3_l=pd.read_csv('dataset3.csv')
df4_l=pd.read_csv('dataset4.csv')
df5_l=pd.read_csv('dataset5.csv')
df6_l=pd.read_csv('dataset6.csv')
df7_l=pd.read_csv('dataset7.csv')
df8_l=pd.read_csv('dataset8.csv')
df9_l=pd.read_csv('dataset9.csv')
df10_l=pd.read_csv('dataset10.csv')
df11_l=pd.read_csv('dataset11.csv')
df12_l=pd.read_csv('dataset12.csv')
df13_l=pd.read_csv('dataset13.csv')
df14_l=pd.read_csv('dataset14.csv')
df15_l=pd.read_csv('dataset15.csv')

df_l=df1_l.append(df2_l.append(df3_l.append(df4_l.append(df5_l.append(df6_l.append(df7_l.append(df8_l.append(df9_l.append(df10_l.append(df11_l.append(df12_l.append(df13_l.append(df14_l.append(df15_l, ignore_index=True), ignore_index=True), ignore_index=True),ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True ), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
df_l['detected_activity']='lying'

###sitting
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\sitting")
df1_s=pd.read_csv('dataset1.csv')
df2_s=pd.read_csv('dataset2.csv')
df3_s=pd.read_csv('dataset3.csv')
df4_s=pd.read_csv('dataset4.csv')
df5_s=pd.read_csv('dataset5.csv')
df6_s=pd.read_csv('dataset6.csv')
df7_s=pd.read_csv('dataset7.csv')
df8_s=pd.read_csv('dataset8.csv')
df9_s=pd.read_csv('dataset9.csv')
df10_s=pd.read_csv('dataset10.csv')
df11_s=pd.read_csv('dataset11.csv')
df12_s=pd.read_csv('dataset12.csv')
df13_s=pd.read_csv('dataset13.csv')
df14_s=pd.read_csv('dataset14.csv')
df15_s=pd.read_csv('dataset15.csv')

df_s=df1_s.append(df2_s.append(df3_s.append(df4_s.append(df5_s.append(df6_s.append(df7_s.append(df8_s.append(df9_s.append(df10_s.append(df11_s.append(df12_s.append(df13_s.append(df14_s.append(df15_s, ignore_index=True), ignore_index=True), ignore_index=True),ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True ), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
df_s['detected_activity']='sitting'


###stading
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\standing")
df1_st=pd.read_csv('dataset1.csv')
df2_st=pd.read_csv('dataset2.csv')
df3_st=pd.read_csv('dataset3.csv')
df4_st=pd.read_csv('dataset4.csv')
df5_st=pd.read_csv('dataset5.csv')
df6_st=pd.read_csv('dataset6.csv')
df7_st=pd.read_csv('dataset7.csv')
df8_st=pd.read_csv('dataset8.csv')
df9_st=pd.read_csv('dataset9.csv')
df10_st=pd.read_csv('dataset10.csv')
df11_st=pd.read_csv('dataset11.csv')
df12_st=pd.read_csv('dataset12.csv')
df13_st=pd.read_csv('dataset13.csv')
df14_st=pd.read_csv('dataset14.csv')
df15_st=pd.read_csv('dataset15.csv')

df_st=df1_st.append(df2_st.append(df3_st.append(df4_st.append(df5_st.append(df6_st.append(df7_st.append(df8_st.append(df9_st.append(df10_st.append(df11_st.append(df12_st.append(df13_st.append(df14_st.append(df15_st, ignore_index=True), ignore_index=True), ignore_index=True),ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True ), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
df_st['detected_activity']='standing'



###walking
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\walking")
df1_w=pd.read_csv('dataset1.csv')
df2_w=pd.read_csv('dataset2.csv')
df3_w=pd.read_csv('dataset3.csv')
df4_w=pd.read_csv('dataset4.csv')
df5_w=pd.read_csv('dataset5.csv')
df6_w=pd.read_csv('dataset6.csv')
df7_w=pd.read_csv('dataset7.csv')
df8_w=pd.read_csv('dataset8.csv')
df9_w=pd.read_csv('dataset9.csv')
df10_w=pd.read_csv('dataset10.csv')
df11_w=pd.read_csv('dataset11.csv')
df12_w=pd.read_csv('dataset12.csv')
df13_w=pd.read_csv('dataset13.csv')
df14_w=pd.read_csv('dataset14.csv')
df15_w=pd.read_csv('dataset15.csv')

df_w=df1_w.append(df2_w.append(df3_w.append(df4_w.append(df5_w.append(df6_w.append(df7_w.append(df8_w.append(df9_w.append(df10_w.append(df11_w.append(df12_w.append(df13_w.append(df14_w.append(df15_w, ignore_index=True), ignore_index=True), ignore_index=True),ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True ), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
df_w['detected_activity']='walking'

###----- final data frame
os.chdir(r"C:\Users\Kian\Desktop\project_dm\AReM\bending1")

df1=df_b1.append(df_b2.append(df_c.append(df_l)))
df=df1.append(df_s.append(df_st.append(df_w)))

###---checking missing number
df.isnull().sum() #no missing data!! wow



###-----encoding and labling
df.detected_activity = df.detected_activity.map({'bending_1':1,'bending_2':2,'cycling':3, 'lying': 4, 'sitting':5, 'standing':6, 'walking':7})
#dataset=pd.get_dummies(df, columns=['detected_activity'], drop_first=True)


df.corrwith(df.detected_activity)



###--- seperation of dependent and independent
X = df.iloc[:,2:7].values
y = df.iloc[:,7].values


df_2 = pd.DataFrame(np.concatenate((X,y.reshape(len(y),1)),axis=1))
df_2.astype(float)
corr = df_2.corr()
sns.heatmap(corr,annot=True)

###---- no need for scaling

###----splitting to test and training
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

accuracy_score(y_test,y_pred) #how many prediction matches with the values 
confusion_matrix(y_test, y_pred)
classification_report(y_test,y_pred)

print(classification_report(y_test,y_pred))

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

