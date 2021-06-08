import tensorflow as tf
from NR.RFBN import RBFLayer, InitCentersRandom
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from NR.RFBN import  kmean_initializer

input_data = np.array([[-1.407547,1  -0.956147,   -0.7971373 ],
 [ 0.1664841  , 0.09258944 , 0.09931423],
 [ 0.48129034 , 0.31504866 , 0.294195  ],
 [ 0.63869345 , 0.88708675 , 0.9567896 ]])
test_input = tf.constant(input_data, dtype=float)
init =  kmean_initializer(input_data)
rbflayer =  RBFLayer(3,initializer=init,p=2,  name='test')(test_input)
tf.print(rbflayer)

#x = np.array([[1,10],[2,15],[3,25]])
#normalizer = Normalization()
#normalizer.adapt(x)
#print(normalizer.mean.numpy(), '  ', normalizer.variance.numpy())
#normalized = normalizer(x).numpy()
#print(normalized)


