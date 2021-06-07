import utils.datapartitional as util
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
if __name__ == '__main__':
    features_columns = ['unique_operators', 'halstead_vocabulary']
    class_column = 'data_set'
    data_set =  util.concat_datasets_files(
        ['./test_data/ar3.csv', './test_data/ar4.csv', './test_data/ar5.csv', './test_data/ar6.csv'])

    data_set = util.concat_datasets(util.create_balanced_buckets(data_set,'defects'))

    x = data_set['unique_operators']
    y = data_set['halstead_vocabulary']
    c = data_set['defects']
    colors = ['red', 'green']

    test_data_set = pd.read_csv('./test_data/ar1.csv', index_col=False)

    fig1 = plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=c, cmap=matplotlib.colors.ListedColormap(colors))
    cb = plt.colorbar()
    loc = np.arange(0, max(c), max(c) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)

    x2 = test_data_set['unique_operators']
    y2 = test_data_set['halstead_vocabulary']
    c2 = test_data_set['defects']

    fig2 = plt.figure(figsize=(8, 8))
    plt.scatter(x2, y2, c=c2, cmap=matplotlib.colors.ListedColormap(colors))
    cb2 = plt.colorbar()
    loc2 = np.arange(0, max(c2), max(c2) / float(len(colors)))
    cb2.set_ticks(loc2)
    cb2.set_ticklabels(colors)

    plt.show()