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
                                       initializer=self.initializer,
                                       trainable=False
                                      )
        self.training_weights = self.add_weight(name='training_weights',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=RandomUniform(0.0, 1.0),
                                       trainable=True
                                       )


    #def call( self, inputs ):
    #    dom = diff_matrix(self.centers, p= self.p, alpha=self.alpha) + tf.constant(10**-8)
    #    C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
    #    H = tf.transpose(C - tf.transpose(inputs))  # matrix of differences
    #    r = tf.exp( tf.math.reduce_sum(H ** 2,1) / dom)
    #    return r
    def call( self, inputs ):
        dom = diff_matrix(self.centers, p= self.p, alpha=self.alpha)
        dom =  2 * ((dom * (self.alpha / self.p)) ** 2)
        distance = euclidean_distance(inputs, self.centers)
        r = tf.exp( distance **2 / dom)
        return tf.matmul( r , self.training_weights)

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
    cols = centers.shape[0]
    diffs = []
    for k in range(cols):
        removed = tf.concat([centers[:k,:], centers[k+1:,:]],0)
        diff = euclidean_distance( tf.expand_dims (centers[k,:],0), removed)
        diff = tf.squeeze(diff)
        if len(diff.shape) > 0 and diff.shape[0] > 1:
            onerowsorted = tf.sort(diff)
        else:
            onerowsorted = tf.reshape(diff, shape=(1,))
        diffs.append(tf.slice(onerowsorted,[0,],[p,]))
    conc = tf.reshape(tf.concat(diffs,0),shape=(-1,p))
    result = tf.reduce_sum(conc, axis=1)
    return result


def euclidean_distance(input_data,centers):
    na = tf.reduce_sum(tf.square(input_data), 1)
    nb = tf.reduce_sum(tf.square(centers), 1)
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    r = tf.sqrt(tf.maximum(na - 2 * tf.matmul(input_data, centers, False, True) + nb, 0.0))
    return r
