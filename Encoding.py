# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:28:40 2019

@author: mfatemeh
"""

import pandas as pd
df= pd.read_csv('titanic_train.csv')

df.head()

############################################DUMMY VARIABLE ENCODING OF ONE VARIABLE
dummies= pd.get_dummies(df.Embarked)

# the inplace will do the change inside the same variables
#in order to not adding a new variables
dummies.drop('C', axis=1, inplace=True)

#removing a variable and add it to the data frame

dataset=df.drop('Embarked', axis=1)

dataset=pd.concat([dummies, dataset], axis=1)

##########################################DUMMY VARIABLE ENCODINF OF MULTIPLE VARIABLES
###################we want to convert multiple columns at once and remove the unnecessary column
#################### column we use the below code
dataset_2=pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


############################################LABEL eNCODING USED FOR ORDINAL VARIABLES
##### in order to convert the ordinal variable
##### in order to copy by refrence not by value we need to use the below command.. we don't want to change the df
dataset_3 = df.copy(deep=True)
dataset_3.Embarked=dataset_3.Embarked.map({'C':1, 'Q':2, 'S':3})
#changing in place.. the inplace arg doesn't work
dataset_3.Sex= dataset_3.Sex.map({'male':0, 'female':1})

###########################################lOOK UP FREQUENCIEES OF USINQ VALUES IN A VARIABLE
df.Embarked.value_counts()


#########################################MISSING HANDLING
df.Embarked.isnull().sum()

# to find the most frequent value in Embark; beside count_value we can use mode but pandas mean and var is for numeric only
mode=df.Embarked.mode()
#mode output is a series because it can be more than one but if we select only the first it is the 
mode2=df.Embarked.mode().iloc[0]

#we can not run belows on categorical
df.Age.mean()
df.Age.median()
df.Age.var()



