# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:49:35 2019

@author: gsaikia
"""

# Importing the libraries
import pandas as pd

# import the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Data Preprocessing
transactions = []
for i in range(0, len(dataset)):
    transaction=[]
    for j in range(0, len(dataset.iloc[0])):
        item=str(dataset.values[i,j])
        if item != 'nan':
            transaction.append(item)
        else:
            break
    transactions.append(transaction)


# Training Apriori on the dataset
from apriori_lib import apriori
rules = apriori(transactions, min_support = 0.005, min_confidence = 0.15, min_lift = 3,
                max_length =4)

# Visualising the results
results = list(rules)
print(len(results))

print(results)
