import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
import numpy as np


class RBFLayer(Layer):


    def __init__(self, output_dim, initializer=None, alpha = 0.5, p=1 , kernel_regularization=None, fixed_centers= True,  **kwargs):
        self.kernel_regularization = kernel_regularization
        self.output_dim = output_dim
        self.init_alpha = alpha
        self.p = p
        self.fixed_centers = fixed_centers
        if initializer:
            self.initializer = initializer
        else:
            self.initializer =RandomUniform(0.0, 1.0)
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       dtype=tf.dtypes.double,
                                       regularizer=self.kernel_regularization,
                                       trainable=not self.fixed_centers)
        self.alpha = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_alpha),
                                     dtype=tf.dtypes.double,
                                     regularizer=self.kernel_regularization,
                                     trainable=not self.fixed_centers)
        if self.fixed_centers:
            self.training_weights = self.add_weight(name='training_weights',
                                         shape=(self.output_dim, input_shape[1]),
                                         initializer=RandomUniform(),
                                         regularizer=self.kernel_regularization,
                                         dtype=tf.dtypes.double,
                                         trainable=True)




    def call( self, inputs ):
        center_diff = diff_matrix(self.centers, p=self.p)
        dom =  2 * ((center_diff * (self.alpha / self.p)) ** 2)
        distance = euclidean_distance(inputs, self.centers)
        power = distance **2 / dom
        r = tf.exp(power)
        if self.fixed_centers:
            r = tf.matmul(r, self.training_weights)
        return r

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def diff_matrix( centers, p = 1 ):
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
