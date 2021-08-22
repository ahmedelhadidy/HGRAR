from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class Normalizer(Layer):

    def __init__(self, factor, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.factor = tf.Variable(name='factor', initial_value=factor, trainable=False, dtype=float)


    def build(self, input_shape):
        super(Normalizer, self).build(input_shape)

    def call( self, x ):
        input= tf.cast (x, dtype=float)
        max = tf.reduce_max(input, axis=0)
        min = tf.reduce_min(input, axis=0)
        max_minus_minimum = max - min
        return self.factor * ((input - min) / max_minus_minimum)

    def compute_output_shape(self, input_shape):
        return input_shape

if __name__ == '__main__':
    arr = np.array([[1,2,3],
                    [4,7,9],
                    [11,13,20]
                   ])
    n= Normalizer(factor=1)
    print(n(arr))




