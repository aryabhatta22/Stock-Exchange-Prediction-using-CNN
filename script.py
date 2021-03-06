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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from sklearn.metrics import confusion_matrix


# ---------------------- Loading Dataset from web ----------------------

def loadDataset():
    print(">>> Loading data")
    start = datetime.datetime(2000, 1,1)
    end = datetime.datetime(2020, 1,1)
    df = web.DataReader('RELIANCE.NS', 'yahoo', start, end )
    # getting rid of Date column
    df = df.reset_index()
    df = df.drop(columns=['Date'])
    # check for null values
    df.isnull().any()
    # Normalizaing data
    sc = MinMaxScaler()
    prices = sc.fit_transform(df)
    # Matrix alottment
    profit = prices[:,3]- prices[:, 2]
    profit = profit.reshape(-1,1)
    return prices, profit


# ---------------------- Creating Synthetic images & Dataset ----------------------


"""
def createImages(X, y):
    i=0
    UpImageNo = 1
    DownImageNo = 1
    my_path = os.getcwd()
                # Saving heatmap
    plt.figure()
    plt.ion()
    #Remove the visibility of axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    fig1 = plt.gcf()
    fig1.patch.set_visible(False)       #Remove the visibility of frame in the graph
    if(y[i] == 1):
        my_file = 'UpGraph'+str(UpImageNo)+'.png'
        Hmap = sns.heatmap(CorrMatrix, cmap='coolwarm', cbar =False)
        #plt.plot(array,X_train[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path,'Dataset/Up/'+my_file), transparent= True)
        plt.pause(0.01)
        UpImageNo = UpImageNo+1
    else:
        my_file = 'DownGraph'+str(DownImageNo)+'.png'
        Hmap = sns.heatmap(CorrMatrix, cmap='coolwarm', cbar =False)
        #plt.plot(array,X_train[i:i+Sample_size,3])
        fig1.savefig(os.path.join(my_path, 'Dataset/Down/'+my_file), transparent= True)
        DownImageNo = DownImageNo+1
        plt.pause(0.001)
    i+=1
    
"""

def createDatset(prices, profit):
    print(">>>> Creating data")
    i=0
    Samplesize = 20
    X = np.empty((0,20,20), dtype = float)
    y = np.empty((0,), dtype = int)    
    while((i+Samplesize) < len(prices)):
        print("i: ",i)
        CorrMatrix = np.corrcoef(prices[i: i+Samplesize][:])
        X = np.insert(X, i, CorrMatrix.reshape(1,20,20), axis = 0)
        if(profit[i:i+Samplesize][0].sum(axis = 0) > 0):
            y = np.insert(y, i, 1, axis = 0)
        else:
            y = np.insert(y, i, 0, axis = 0)
        i+=1
    return X, y

# ---------------------- Splitting data into taining and test set ----------------------
        
def splitDataset(X, y):
    print(">>>> Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.reshape((X_train.shape[0], 20, 20, 1))
    X_test = X_test.reshape((X_test.shape[0], 20, 20, 1))
    print("training set: ",X_train.shape)
    print("test set: ",X_test.shape)
    return X_train, X_test, y_train, y_test

# ---------------------- Model Architecture ----------------------
    
def model():
    classifier = Sequential()
    classifier.add(Conv2D(32,(3, 3),activation='relu', input_shape=(20, 20, 1)))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    classifier.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    classifier.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
    classifier.add(Dense(1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# ---------------------- Model Evaluation ----------------------

def evaluateModel(X_train, X_test, y_train, y_test):
    print(">>>> Executing model")
    classifier = model()
    history = classifier.fit(X_train, y_train, epochs=50, validation_split=0.2,  batch_size=64, validation_data=(X_test, y_test), verbose=0)
    eval_model = classifier.evaluate(X_train, y_train)
    y_pred=classifier.predict(X_test)
    y_pred =(y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    return history, eval_model, cm

def modelStats(history, eval_model, cm):
    print('---------- Model Performance ----------')
    print('Loss: ', eval_model[0])
    print('Accuracy: ', eval_model[1])
    print('Confusion matrix: ')
    print(cm)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Model Accuracy.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Model Loss.png')
    plt.show()

# ---------------------- Running Pipeline----------------------
    
def RunModel():
    print(">>>> Running model")
    (prices, profit) = loadDataset()
    print(prices.shape)
    print(profit.shape)
    (X, y) = createDatset(prices, profit)
    (X_train, X_test, y_train, y_test) = splitDataset(X, y)
    (history, eval_model, cm) = evaluateModel(X_train, X_test, y_train, y_test)
    modelStats(history, eval_model, cm)
    
RunModel()