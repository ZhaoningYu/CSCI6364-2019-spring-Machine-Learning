#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:02:22 2019

@author: yzn
"""

# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

test = pd.read_csv('../Datasets/test.csv')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)

# Fitting Random Forest Classification to the Training set
# Create your classifier here
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Validation set results
y_pred = classifier.predict(X_validation)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_validation, y_pred)

# Calculate the accuracy_score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_validation,y_pred))

# Predicting the Test set

test_pred = classifier.predict(test)

result_test_pred = pd.DataFrame(test_pred, columns=['Label'])
result_test_pred['ImageId'] = test.index + 1

result_test_pred[['ImageId', 'Label']].to_csv('../Datasets/result_submission.csv', index=False)