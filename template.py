# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:46:49 2017

@author: Vijay
"""

# Assignment 2 : Implementation of a fully connected Nueral Network from scratch 
import random
import numpy as np
import pandas as pd

def get_feature_matrix(file_path):
    datafile = open(file_path, 'rb')        # open file with name
    datareader = pd.read_csv(datafile,header=0,usecols =['s1','c1','s2','c2','s3','c3','s4','c4','s5','c5'],delimiter=',')
    a = datareader.values                   # store the matrix a
    return a
    
def get_output(file_path):
    datafile = open(file_path, 'rb')        # open file with name
    datareader = pd.read_csv(datafile,header=0, usecols=['class'])
    a = datareader.values                   # store as array/matrix
    return a
        
features_train = get_feature_matrix('train.csv')
labels_train = get_output('train.csv')
features_train_norm = (features_train - features_train.mean())/(features_train.max() - features_train.min()) # Normalized data
#d_test = get_feature_matrix('test.csv')
print(features_train_norm)


learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
num_hidden_layers = [1, 2, 3, 4, 5]
regularizer_lambda = [100, 10, 1, 1e-1, 1e-2]

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=None, shuffle=True)
kf.get_n_splits(features_train_norm)
print(kf)  
for train_index, test_index in kf.split(features_train_norm):
   X_train, X_test = features_train[train_index], features_train[test_index]
   y_train, y_test = labels_train[train_index], labels_train[test_index]
   
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='sgd', alpha=100,hidden_layer_sizes=(1), learning_rate_init = 1e-1, random_state=1)
clf.fit(X_train,y_train.ravel())
pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,pred)
#print(ac)
