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
    
# Assigns each entry to a non-full partition
def assign(partitions, k, size, seed=None):
    if seed is not None:
        random.Random(seed)
    x = random.randint(0,k-1)
    while(len(partitions[x]) >= size):
        x = random.randint(0,k-1)
    return x

# Divides data set into k partitions
def partition(dataSet, k, seed=None):
    size = np.ceil(len(dataSet)/float(k))
    partitions = [[] for i in range(k)]
    j = 0
    
    for entry in dataSet:
        x = assign(partitions, k, size, seed) 
        partitions[x].append(entry)

    return partitions
    
features_train = get_feature_matrix('train.csv')
labels_train = get_output('train.csv')
features_train_norm = (features_train - features_train.mean())/(features_train.max() - features_train.min()) # Normalized data
#d_test = get_feature_matrix('test.csv')
print(d_train_norm)
part = partition(d_train_norm, 80000)

learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
num_hidden_layers = [1, 2, 3, 4, 5]
regularizer_lambda = [100, 10, 1, 1e-1, 1e-2]

# Function for K fold cross validation
def kfoldCV(classifier, features, k, seed=None):
    partitions = partition(features, k, seed)
    errors = list()
        
    # Run the algorithm k times, record error each time
    for i in range(k):
        trainingSet = list()
        for j in range(k):
            if j != i:
                trainingSet.append(partitions[j])

        # flatten training set
        trainingSet = [item for entry in trainingSet for item in entry]
        testSet = partitions[i]
        
        # Train and classify model
        trainedClassifier = train(classifier, trainingSet)
        errors.append(classify(classifier, testSet))
        
    # Compute statistics
    mean = sum(errors)/k
    variance = sum([(error - mean)**2 for error in errors])/(k)
    standardDeviation = variance**.5
    confidenceInterval = (mean - 1.96*standardDeviation, mean + 1.96*standardDeviation)
 
    _output("\t\tMean = {0:.2f} \n\t\tVariance = {1:.4f} \n\t\tStandard Devation = {2:.3f} \n\t\t95% Confidence interval: [{3:.2f}, {4:.2f}]"\
            .format(mean, variance, standardDeviation, confidenceInterval[0], confidenceInterval[1]))

    return (errors, mean, variance, confidenceInterval, k)

