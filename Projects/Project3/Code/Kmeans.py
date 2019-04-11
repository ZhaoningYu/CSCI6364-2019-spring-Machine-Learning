#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:56:09 2019

@author: yzn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/Human/train.csv')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

test = pd.read_csv('../Datasets/Human/test.csv')
test_x = test.iloc[:, 1:].values

'''
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
y = labelencoder_x.fit_transform(y)
'''

# Transfer 6 labels to 2 labels 0 - move, 1 - not move
#y = y.tolist()
labels = y.copy()
for i in range(len(labels)):
    if (labels[i] == 'STANDING' or labels[i] == 'SITTING' or labels[i] == 'LAYING'):
        labels[i] = 1
    else:
        labels[i] = 0
labels = np.array(labels.astype(int))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test_x = sc.transform(test_x)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
X = pca.fit_transform(X)
test_x = pca.transform(test_x)
explained_variance = pca.explained_variance_ratio_

# Using the elbow method to find the optimal numbers of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('The number of cluster')
plt.ylabel('wcss')
plt.show()

# Applying the K-means algorithm with 2 clusters to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Applying the K-means algorithm with 6 clusters to the dataset
# kmeans_6 = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
# y_kmeans_6 = kmeans_6.fit_predict(X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, y_kmeans)

# Calculate the accuracy_score
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, y_kmeans))















