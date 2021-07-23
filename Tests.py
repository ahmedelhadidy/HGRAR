import utils.datapartitional as util
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
if __name__ == '__main__':
    arr_one = np.ones(shape=(10,))
    arr_5 = np.full(fill_value=0.5, shape=(10,))
    arr_3 = np.full(fill_value=0.3, shape=(10,))
    ff = abs(arr_3 - arr_5)
    print(np.sum(ff))