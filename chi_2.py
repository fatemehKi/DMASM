# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:04:22 2019

@author: gsaikia
"""

import pandas as pd

df = pd.read_csv('titanic_train.csv')

df.isnull().sum()

observed_contigency_table = pd.crosstab(df.Survived,df.Sex)

from scipy.stats.contingency import chi2_contingency

chi_2, p_val, dof, expected_contingency_table = chi2_contingency(observed_contigency_table)

alpha = 0.05

if p_val < alpha:
    print('The variables are correlated at signicance level',alpha)
else:
    print('We fail to reject that the variables are independent at signicance level',alpha)













'''
c_t = pd.DataFrame([[250,200],[50,1000]],columns=['Plays Chess','Doesnt Play Chess'],
                   index=['Likes Science Fiction','Doesnt Like Science Fiction'])

chi2_contingency(c_t)
'''













