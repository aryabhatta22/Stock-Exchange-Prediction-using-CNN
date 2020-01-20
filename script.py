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
Y =X[:, 3:4]-X[:, 0:1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = False )

# ---------------------- Creating Synthetic images ----------------------
import os
import numpy as np
import matplotlib.pyplot as plt

my_path = os.getcwd()       # Path of current working directory
Sample_size = 20
array = np.arange(Sample_size)

                        # Creating trainig image set

i=0
UpImageNo = 1
DownImageNo = 1

while i+Sample_size <= len(X_train[:,0]):
    plt.figure()
    #Remove the visibility of axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    fig1 = plt.gcf()
    fig1.patch.set_visible(False)       #Remove the visibility of frame in the graph
    print(i)
    if Y_train[i:i+Sample_size,0].sum() >=0:
        my_file = 'UpGraph'+str(UpImageNo)+'.png'
        plt.plot(array,X_train[i:i+Sample_size,0])
        plt.plot(array,X_train[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path,'Dataset/Training/Up/'+my_file), transparent= True)
        UpImageNo = UpImageNo+1
    else:
        my_file = 'DownGraph'+str(DownImageNo)+'.png'
        plt.plot(array,X_train[i:i+Sample_size,0])
        plt.plot(array,X_train[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path, 'Dataset/Training/Down/'+my_file), transparent= True)
        DownImageNo = DownImageNo+1
    i=i+Sample_size

                        # Creating test image set

i=0
UpImageNo = 1
DownImageNo = 1

while i+Sample_size <= len(X_test[:,0]):
    plt.figure()
    #Remove the visibility of axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    fig1 = plt.gcf()
    fig1.patch.set_visible(False)       #Remove the visibility of frame in the graph
    print(i)
    if Y_test[i:i+Sample_size,0].sum() >=0:
        my_file = 'UpGraph'+str(UpImageNo)+'.png'
        plt.plot(array,X_test[i:i+Sample_size,0])
        plt.plot(array,X_test[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path,'Dataset/Test/Up/'+my_file), transparent= True)
        UpImageNo = UpImageNo+1
    else:
        my_file = 'DownGraph'+str(DownImageNo)+'.png'
        plt.plot(array,X_test[i:i+Sample_size,0])
        plt.plot(array,X_test[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path, 'Dataset/Test/Down/'+my_file), transparent= True)
        DownImageNo = DownImageNo+1
    i=i+Sample_size


# ---------------------- CNN ----------------------