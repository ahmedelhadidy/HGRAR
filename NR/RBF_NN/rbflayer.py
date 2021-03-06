from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import tensorflow as tf
import sys

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

    def __init__(self, output_dim, initializer=None, betas=1.0, kernel_regularization=None, fixed_centers= True, **kwargs):
        self.kernel_regularization = kernel_regularization
        self.output_dim = output_dim
        self.init_betas = betas
        self.fixed_centers = fixed_centers
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       regularizer=self.kernel_regularization,
                                       dtype=tf.dtypes.double,
                                       trainable=not self.fixed_centers)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_betas),
                                     regularizer=self.kernel_regularization,
                                     dtype=tf.dtypes.double,
                                     trainable=not self.fixed_centers)
        if self.fixed_centers:
            self.training_weights = self.add_weight(name='training_weights',
                                         shape=(self.output_dim, input_shape[1]),
                                         initializer=RandomUniform(),
                                         regularizer=self.kernel_regularization,
                                         dtype=tf.dtypes.double,
                                         trainable=True)


        super(RBFLayer, self).build(input_shape)

    # def call(self, x):
    #     C = K.expand_dims(self.centers)
    #     H = K.transpose(C-K.transpose(x))
    #     power = -self.betas * K.sum(H**2, axis=1)
    #     r = K.exp(power)
    #     if self.fixed_centers:
    #         r = tf.matmul(r, self.training_weights)
    #     return r

    def call( self, x ):
        C = K.expand_dims(self.centers)
        H = K.transpose(C - K.transpose(x))
        power = - 0.5 * K.sum((H / self.betas)**2, axis=1)
        r = K.exp(power)
        if self.fixed_centers:
            r = tf.matmul(r, self.training_weights)
        return r


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))