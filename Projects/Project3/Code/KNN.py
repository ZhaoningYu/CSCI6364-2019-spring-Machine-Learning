#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:43:26 2019

@author: yzn
"""

# -*- coding: utf-8 -*-
'''
Use KNN algorithm to analyze digit-recognizer dataset
Copyright by Zhaoning Yu 02/05/2019

'''

# Import the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# Calculate running time
import time
start = time.time()

# Import datasets
train = pd.read_csv('../Datasets/train.csv')
features_train = train.iloc[:, 1:].values
result_train = train.iloc[:, 0].values

test = pd.read_csv('../Datasets/test.csv')
features_test = test.iloc[:, :].values
#result_test = test.iloc[:, 8].values


# Feature scaling
# from sklearn.preprocessing import StandardScaler
# ss_features = StandardScaler()
# features_train = ss_features.fit_transform(features_train)
# features_test = ss_features.transform(features_test)

# Splitting the training dataset into training set and validate set
from sklearn.model_selection import train_test_split
features_training, features_validate, result_training, result_validate = train_test_split(features_train, result_train, test_size = 0.25, random_state = 0)

# Fit classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(features_training, result_training)

# Predicting the validate set results
result_pred = classifier.predict(features_validate)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm_validate = confusion_matrix(result_validate, result_pred)

# Calculate the accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(result_validate,result_pred))

# Predicting the test set results
test_pred = classifier.predict(features_test)

result_test_pred = pd.DataFrame(test_pred, columns=['Label'])
result_test_pred['ImageId'] = test.index + 1

result_test_pred[['ImageId', 'Label']].to_csv('../Datasets/result_submission_knn.csv', index=False)

# Print running time
end = time.time()
print (end - start)
