"""
Name: Alek Michael
K-NN 
"""

import NoiseGeneration as ng
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import mean_squared_error as mse
import time

# Obtain split data
X_train, X_test, Y_train, Y_test = ng.splitVariants()

start_time = time.time()

# init kNN
knn = kNN(
        n_neighbors = 10
)

knn.fit(X_train, Y_train)

predictions = knn.predict(X_test)

error = mse(Y_test, predictions)

print('k-NN error ', error)

print('Time (sec): ', time.time() - start_time)