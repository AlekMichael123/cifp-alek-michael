"""
Name: Alek Michael
Feature Selection using support vector classifier 
"""

import NoiseGeneration as ng
import numpy as np
import FinalProjectSVM as SVM
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot
import matplotlib.pyplot as plt

svm = SVM.classifier

sfs = SFS(
        svm,
        k_features = 20,
        forward = True,
        floating = False,
        scoring = 'accuracy',
        cv = 0
)

# Obtain split data
X_train, X_test, Y_train, Y_test = ng.splitVariants()

X = np.concatenate((X_train, X_test))
Y = np.concatenate((Y_train, Y_test))

sfs.fit_transform(X, Y)

print(sfs.k_feature_names_)

fig = plot(sfs.get_metric_dict(),
           kind = 'std_dev',
           figsize = (6, 4)
)

plt.ylim([0.8, 1])
plt.grid()
plt.show()