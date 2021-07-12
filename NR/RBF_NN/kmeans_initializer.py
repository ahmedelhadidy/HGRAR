from tensorflow.keras.initializers import Initializer
from sklearn.cluster import KMeans
import numpy as np


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


class InitCentersKMeans2(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X_cls1, X_cls2, max_iter=100):
        self.X_cls1 = X_cls1
        self.X_cls2 = X_cls2
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[0] % 2 == 0
        assert shape[1:] == self.X_cls1.shape[1:]
        assert shape[1:] == self.X_cls2.shape[1:]

        n_centers = shape[0]
        n_centers_per_class = int(n_centers / 2)

        km_cl1 = KMeans(n_clusters=n_centers_per_class, max_iter=self.max_iter, verbose=0)
        km_cl1.fit(self.X_cls1)

        km_cl2 = KMeans(n_clusters=n_centers_per_class, max_iter=self.max_iter, verbose=0)
        km_cl2.fit(self.X_cls2)
        return np.concatenate((km_cl1.cluster_centers_, km_cl2.cluster_centers_))