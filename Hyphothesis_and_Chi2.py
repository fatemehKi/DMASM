# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:02:49 2019

@author: mfatemeh
"""

import pandas as pd

df =pd.read_csv('titanic_train.csv')
df.isnull().sum()

observed_contigency_table = pd.crosstab(df.Survived, df.Pclass)

from scipy.stats.contingency import chi2_contingency

chi_2, p_val, dof, expected_contigency_table = chi2_contingency(observed_contigency_table)
#the frredom ddegree is 2 because 3-1 =2 * 1 =2
#p_value is vey low therefore the H_0 is regected meaning not(they are independent); therefore, they are dependent(correlated)


observed_contigency_table2 = pd.crosstab(df.Survived, df.Sex)

chi_2_G, p_val, dof_G, expected_contigency_table_G = chi2_contingency(observed_contigency_table2)
# if p_value is less than alpha (ass)

alpha=0.05
if p_val< alpha:
    print('The varianles are correlated at significant level', alpha)
else:
    print('The variables are independent at signifiacant level', alpha)
