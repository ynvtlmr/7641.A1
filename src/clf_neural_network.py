from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
import numpy as np


class MLP(object):

    def __init__(self):
        """ Construct the multi-layer perceptron classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline([
            ('Scale', StandardScaler()),
            ('MLP', MLPClassifier(max_iter=100, early_stopping=False))
        ])

        # set up parameter grid for parameters to search over
        alphas = [10 ** -exp for exp in np.arange(0.5, 3, 0.25)]
        d = 25
        hidden_layer_size = [(h,) * l for l in [1, 2]
                             for h in [d // 2, d, d * 2]]

        self.params = {'MLP__activation': ['relu'],  # 'logistic',
                       'MLP__alpha': alphas,
                       'MLP__solver': ['adam'],
                       'MLP__hidden_layer_sizes': hidden_layer_size,
                       'MLP__learning_rate': ['constant']
                       }
