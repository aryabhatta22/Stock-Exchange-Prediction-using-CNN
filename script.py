# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:26:59 2020

@author: tarun
"""

# ---------------------- Data preprocessing ----------------------

import pandas as pd

dataset = pd.read_csv('RELIANCE.NS.csv')
X = dataset.iloc[:, 1: 5].values  

X = X[X[:,0] != -1]
Y = X[:, 3:4]-X[:, 0:1] >0

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )

