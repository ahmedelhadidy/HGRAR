from tensorflow.keras.initializers import Initializer
import numpy as np


class InitFromFile(Initializer):
    """ Initialize the weights by loading from file.

    # Arguments
        filename: name of file, should by .npy file
    """
    def __init__(self, filename):
        self.filename = filename
        super().__init__()

    def __call__(self, shape, dtype=None):
        with open(self.filename, "rb") as f:
            X = np.load(f, allow_pickle=True) # fails without allow_pickle
        assert tuple(shape) == tuple(X.shape)
        return X

    def get_config(self):
        return {
            'filename': self.filename
        }


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]  # check dimension

        # np.random.randint returns ints from [low, high) !
        idx = np.random.randint(self.X.shape[0], size=shape[0])

        return self.X[idx, :]