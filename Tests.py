import utils.datapartitional as util
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from keras import backend as K
import math
if __name__ == '__main__':
    arr = np.array([[-72.7757797, 102.527214, -165.703552, -146.717346],
                     [-0.68203938, 1.7974422, -0.123710774, -1.52991772],
                     [-3.15956116, 0.170159593, -2.96497917, -0.0195852909],
                     [-0.239001513, 4.88710785, -1.15092349, -5.38040161],
                     [-0.761349618, 1.67759407, -0.177857444, -1.39482045],
                     [-8.22487354, 19.8075466, -21.5040684, -26.2860794],
                     [-2.72117352, 0.275216222, -2.35134292, -0.0548172742]])
    r = K.exp(arr)
    print(r)
    r=K.log(arr)
    print(r)