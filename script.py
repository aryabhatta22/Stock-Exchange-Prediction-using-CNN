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

from keras.models import Sequential       
from keras.layers import Convolution2D   
from keras.layers import MaxPooling2D     
from keras.layers import Flatten          
from keras.layers import Dense            

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                         
# ---------------------- Fitting the CNN ----------------------

from keras.preprocessing.image import ImageDataGenerator
                
train_datagen = ImageDataGenerator(
        rescale=1,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1)

training_set= train_datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(64, 64),
        batch_size=5,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'Dataset/Test',
        target_size=(64, 64),
        batch_size=5,
        class_mode='binary')
 
classifier.fit_generator(
        training_set,
        steps_per_epoch=197,
        epochs=10,
        validation_data=test_set,
        validation_steps=49)