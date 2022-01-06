"""
Name: Alek Michael
MLP 
"""

import NoiseGeneration as ng
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error as mse
import time

# Obtain split data
X_train, X_test, Y_train, Y_test = ng.splitVariants()

start_time = time.time()

# init MLP
mlp = MLPClassifier(
        hidden_layer_sizes = [6, 3, 6],
        activation = 'relu',
        max_iter = 1000000,
        solver = 'sgd',
        learning_rate = 'adaptive',
        learning_rate_init = 0.001,
        random_state = 123
)

mlp.fit(X_train, Y_train)

predictions = mlp.predict(X_test)

error = mse(Y_test, predictions)

print('MLP Error: ', error)

print ('Time (sec): ', time.time() - start_time) 