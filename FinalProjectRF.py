"""
Name: Alek Michael
Random Forest
"""

import NoiseGeneration as ng
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse
import time

# Obtain split data
X_train, X_test, Y_train, Y_test = ng.splitVariants()

start_time = time.time()

# init RF
rf = RandomForestClassifier(
        n_estimators = 2000,
        max_depth = None,
        random_state = 0,
)

rf.fit(X_train, Y_train)

predictions = rf.predict(X_test)

error = mse(Y_test, predictions)

print('RF error ', error)

print('Time (sec): ', time.time() - start_time)