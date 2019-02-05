from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


class SVM_RBF(object):

    def __init__(self):
        """ Construct the SVM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('SVMR', SVC())])

        # set up parameter grid for parameters to search over
        self.params = {'SVMR__kernel': ['rbf'],
                       'SVMR__C': np.linspace(0.1, 1.0, num=10),
                       'SVMR__gamma': np.logspace(-9, 1, 15),
                       'SVMR__cache_size': [8000],
                       'SVMR__max_iter': [40000],
                       'SVMR__class_weight': ['balanced']
                       }


class SVM_PLY(object):

    def __init__(self):
        """ Construct the SVM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('SVMP', SVC())])

        # set up parameter grid for parameters to search over
        self.params = {'SVMP__kernel': ['poly'],
                       'SVMP__C': np.linspace(0.1, 1.0, num=10),
                       'SVMP__gamma': np.logspace(-9, 1, 10),
                       'SVMP__cache_size': [8000],
                       'SVMP__max_iter': [40000],
                       'SVMP__degree': np.arange(1, 5, 2),
                       'SVMP__coef0': [0, 1],
                       'SVMP__class_weight': ['balanced']
                       }


class SVM_LIN(object):

    def __init__(self):
        """ Construct the SVM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('SVML', SVC())])

        # set up parameter grid for parameters to search over
        self.params = {'SVML__kernel': ['linear'],
                       'SVML__C': np.linspace(0.1, 1.0, num=10),
                       'SVML__cache_size': [8000],
                       'SVML__max_iter': [40000],
                       'SVML__class_weight': ['balanced']
                       }


class SVM_SIG(object):

    def __init__(self):
        """ Construct the SVM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('SVMS', SVC())])

        # set up parameter grid for parameters to search over
        self.params = {'SVMS__kernel': ['sigmoid'],
                       'SVMS__C': np.linspace(0.1, 1.0, num=10),
                       'SVMS__cache_size': [8000],
                       'SVMS__max_iter': [40000],
                       'SVMS__class_weight': ['balanced'],
                       'SVMS__gamma': np.logspace(-9, 1, 10),
                       'SVMS__coef0': [0, 1],
                       }
