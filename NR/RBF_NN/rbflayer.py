import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer
import numpy as np


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


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.

    # Example

    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```


    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas

    """

    def __init__(self, output_dim, initializer=None, alpha = 0.5, p=1 ,  **kwargs):
        self.output_dim = output_dim
        self.alpha = alpha
        self.p = p
        if initializer:
            self.initializer = initializer
        else:
            self.initializer =RandomUniform(0.0, 1.0)
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer
                                      )

    def call( self, inputs ):
        print('======================A7A=========================')
        dom = diff_matrix(self.centers, p= self.p, alpha=self.alpha)
        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
        H = tf.transpose(C - tf.transpose(inputs))  # matrix of differences
        r = tf.exp( tf.math.reduce_sum(H ** 2,1) / dom)
        return r



    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def diff_matrix( centers, p = 1, alpha = 0.5 ):
    centers_T = tf.transpose(centers)
    cols = centers_T.shape[1]
    diffs = []
    for k in range(cols):
        removed = tf.concat([centers_T[:,:k], centers_T[:,k+1:]],-1)
        diff = tf.abs( tf.expand_dims(centers_T[:,k],-1) - removed)
        onerow = tf.reduce_sum(diff, axis=0)
        onerowsorted = tf.sort(onerow)
        diffs.append(tf.slice(onerowsorted,[0,],[p,]))
    conc = tf.reshape(tf.concat(diffs,0),shape=(-1,p))
    result = tf.reduce_sum(conc, axis=1)
    result = 2 * (result * (alpha / p)) ** 2
    return result
