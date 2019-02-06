
'''
Use KNN algorithm to analyze diabetes dataset
Copyright by Zhaoning Yu 02/05/2019

'''
# Import the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# Calculate running time
import time
start = time.time()

# Import diabetes
diabetes = pd.read_csv('../Datasets/diabetes.csv')
features = diabetes.iloc[:, :-1].values
result = diabetes.iloc[:, 8].values

# Taking care of missing data
# Using Imputer
'''
from sklearn.preprocessing import Imputer 
diabetes_imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
diabetes_imputer = diabetes_imputer.fit(features[:, 1:8])
features[:, 1:8] = diabetes_imputer.transform(features[:, 1:8])
'''

# Using SimpleImputer (new version)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
features_train, features_test, result_train, result_test = train_test_split(features, result, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
ss_features = StandardScaler()
features_train = ss_features.fit_transform(features_train)
features_test = ss_features.transform(features_test)

# Fit classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
classifier.fit(features_train, result_train)

# Predicting the test set results
result_pred = classifier.predict(features_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(result_test, result_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(result_test,result_pred))

# Print running time
end = time.time()
print (end - start)








