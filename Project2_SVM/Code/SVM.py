#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:18:28 2019

@author: yzn
"""

# Classification template

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/breast-cancer-wisconsin.txt', header = None)
dataset[6] = dataset[6].replace('?', 0)
dataset[6] = pd.to_numeric(dataset[6])
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 10].values
print(dataset[6].dtypes)

# Taking care of missing data
# Using Imputer
from sklearn.preprocessing import Imputer 
dataset_imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
dataset_imputer = dataset_imputer.fit(X[:, [5]])
X[:, [5]] = dataset_imputer.transform(X[:, [5]])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
