# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:26:59 2020

@author: tarun
"""

# ---------------------- Data preprocessing ----------------------

import pandas as pd
import pandas_datareader.data as web
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns


start = datetime.datetime(2000, 1,1)
end = datetime.datetime(2020, 1,1)
df = web.DataReader('RELIANCE.NS', 'yahoo', start, end )

# getting rid of Date column
df = df.reset_index()
df = df.drop(columns=['Date'])

# check for null values
df.isnull().any()

# Normalizaing data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
prices = sc.fit_transform(df)

# Matrix alottment

profit = prices[:,3]- prices[:, 2]
profit = profit.reshape(-1,1)


# ---------------------- Creating Synthetic images & Dataset ----------------------

my_path = os.getcwd()       # Path of current working directory
Samplesize = 20

                        # Creating trainig image set

UpImageNo = 1
DownImageNo = 1


i=0
X = np.empty((0,20,20), dtype = float)
Y = np.empty((0,), dtype = int)

while((i+Samplesize) < len(prices)):
    CorrMatrix = np.corrcoef(prices[i: i+Samplesize][:])
    X = np.insert(X, i, CorrMatrix.reshape(1,20,20), axis = 0)
    if(profit[i:i+Samplesize][0].sum(axis = 0) > 0):
        Y = np.insert(Y, i, 1, axis = 0)
    else:
        Y = np.insert(Y, i, 0, axis = 0)
        
                    # Saving heatmap
    plt.figure()
    #Remove the visibility of axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    fig1 = plt.gcf()
    fig1.patch.set_visible(False)       #Remove the visibility of frame in the graph
    if(Y[i] == 1):
        my_file = 'UpGraph'+str(UpImageNo)+'.png'
        sns.heatmap(CorrMatrix, cmap='coolwarm', cbar =False)
        #plt.plot(array,X_train[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path,'Dataset/Up/'+my_file), transparent= True)
        UpImageNo = UpImageNo+1
    else:
        my_file = 'DownGraph'+str(DownImageNo)+'.png'
        sns.heatmap(CorrMatrix, cmap='coolwarm', cbar =False)
        #plt.plot(array,X_train[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path, 'Dataset/Down/'+my_file), transparent= True)
        DownImageNo = DownImageNo+1
    i+=1

