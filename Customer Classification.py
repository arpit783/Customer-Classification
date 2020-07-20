# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:12:23 2020

@author: Hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  9 21:06:12 2020

@author: Hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:53:41 2020

@author: Hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:18:57 2020

@author: Hp
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(penalty = 'l1', max_iter = 50, random_state = 0)
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'max_iter': [50, 100 , 150, 200], 'C': [1, 2, 100, 1000], 'tol': [1e-4, 1e-5, 1e-3, 1e-2], 'penalty': ['l1']}, {'max_iter': [50, 100 , 150, 200], 'C': [1, 2, 100, 1000], 'tol': [1e-4, 1e-5, 1e-3, 1e-2], 'penalty': ['l2']}]
grid_search = GridSearchCV(estimator = regressor, cv = 10, n_jobs = -1, scoring = 'accuracy', param_grid = parameters)
grid_search = grid_search.fit(X_train, Y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1, stop = X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1, stop = X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test Set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()