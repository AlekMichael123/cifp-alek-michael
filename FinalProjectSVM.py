"""
Name: Alek Michael
SVM
"""

import NoiseGeneration as ng
from sklearn import svm
from sklearn.metrics import mean_squared_error as mse
import time

# Obtain split data
X_train, X_test, Y_train, Y_test = ng.splitVariants()

start_time = time.time()

# init SVM
classifier = svm.SVC(gamma = 'auto')

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

error = mse(Y_test, predictions)

print('SVM error ', error)

print('Time (sec): ', time.time() - start_time)