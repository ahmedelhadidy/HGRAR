import tensorflow as tf
from NR.RFBN import RBFLayer, InitCentersRandom
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from NR.RFBN import kmean_initializer
import os.path as path
import csv
from tensorflow.keras.initializers import Initializer



class TestInit(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X):
        self.X = X
        super().__init__()

    def __call__(self, shape, dtype=None):
        return self.X

def test_rbf():

    input_data = np.array([[-0.47681463 ,-0.6508253 ],
                             [ 0.4128518  ,-0.4056642 ],
                             [ 2.3404624  , 3.5441537 ],
                             [-0.18025915 ,-0.32394385],
                             [ 0.11629633 ,-0.48738456],
                             [ 2.043907   , 3.271752  ],
                             [-1.3664811  ,-0.92322654],
                             [ 0.56112957 ,-0.48738456],
                             [-0.47681463 ,-0.7597858 ],
                             [-0.6250924  ,-0.59634507],
                             [-0.92164785 ,-0.814266  ],
                             [-0.92164785 ,-0.2694636 ],
                             [ 1.3025182  , 0.4660197 ],
                             [-0.18025915 ,-0.2694636 ],
                             [ 0.56112957 , 0.32981908],
                             [-0.3285369  ,-0.5146247 ],
                             [-0.6250924  ,-0.5691049 ],
                             [ 0.26457408 , 0.22085859],
                             [ 0.56112957 , 0.35705918],
                             [ 0.56112957 , 0.22085859],
                             [-0.03198141 ,-0.10602287],
                             [-0.7733701  ,-0.8959864 ],
                             [ 0.56112957 ,-0.4056642 ],
                             [-0.18025915 ,-0.5418648 ],
                             [-1.3664811  ,-1.0049468 ],
                             [-1.2182033  ,-0.7597858 ],
                             [-1.3664811  ,-1.032187  ],
                             [ 0.857685   , 0.95634186],
                             [-1.3664811  ,-1.0866672 ],
                             [ 2.6370177  , 1.6101048 ]])

    centers = np.array([[-0.93249744, -0.7717447 ],
       [ 1.4238365 ,  1.5469573 ],
       [5,5]
                        ])

    test_input = tf.constant(input_data, dtype=float)
    init = TestInit(centers)# kmean_initializer(input_data)
    rbflayer = RBFLayer(3, initializer=init, p=2, alpha=0.5,  name='test')(test_input)
    d = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)(rbflayer)
    tf.print(rbflayer)
    tf.print(d)


def test_euclidean():
    a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=float)
    b = tf.constant([[12, 3], [14, 5], [16, 7]], dtype=float)
    r = tf.norm(a - b, ord='euclidean', axis=0)
    print(r)

def test_euclidean_batch():
    input_data = tf.constant(np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1],
                       [2, 2, 2]]), dtype=float)
    centers = tf.constant(np.array([[-0.5, -0.5, -0.5],
                        [0.5, 0.5, 0.5],
                        [1.5, 1.5, 1.5]]), dtype=float)
    na = tf.reduce_sum(tf.square(input_data), 1)
    nb = tf.reduce_sum(tf.square(centers), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    r = tf.sqrt(tf.maximum(na - 2 * tf.matmul(input_data, centers, False, True) + nb, 0.0))
    print(r)


def euclidean_distance(input_data,centers):
    na = tf.reduce_sum(tf.square(input_data), 1)
    nb = tf.reduce_sum(tf.square(centers), 1)
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    r = tf.sqrt(tf.maximum(na - 2 * tf.matmul(input_data, centers, False, True) + nb, 0.0))
    return r

def diff_matrix( centers, p = 1, alpha = 0.5 ):
    cols = centers.shape[0]
    diffs = []
    for k in range(cols):
        removed = tf.concat([centers[:k,:], centers[k+1:,:]],0)
        diff = euclidean_distance( tf.expand_dims (centers[k,:],0), removed)
        diff = tf.squeeze(diff)
        onerowsorted = tf.sort(diff)
        diffs.append(tf.slice(onerowsorted,[0,],[p,]))
    conc = tf.reshape(tf.concat(diffs,0),shape=(-1,p))
    result = tf.reduce_sum(conc, axis=1)
    result = 2 * (result * (alpha / p)) ** 2
    return result

def test_diff_matrix():
    centers = tf.constant(np.array([[-0.5, -0.5, -0.5],
                                    [0.5, 0.5, 0.5],
                                    [1.5, 1.5, 1.5]]), dtype=float)
    d = diff_matrix(centers)


def test_matrix_mul():
    ar1 = tf.constant(np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]), dtype=float)
    ar2 = tf.constant(np.array([[10,10,10],[20,20,20],[30,30,30]]), dtype=float)
    print(tf.matmul(ar1, ar2))

def test_np_split():
    x = np.array([[1,2],[5,7],[8,3],[2,1],[4,4],[22,44],[55,77],[66,23],[45,89],[57,23]])
    y = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0],[1,0]])
    new_x, new_y, (T_x, T_y) , (F_x, F_y) = get(x,y,([0,1],0.1), ([1,0], 0.2))
    x_val = np.concatenate((T_x, F_x), axis=0)
    y_val = np.concatenate((T_y,F_y), axis=0)
    print(new_x, '\n\n', new_y, '\n\n',x_val, '\n\n',y_val )

def get(x, y, *args):
    r = []
    for val, percentage in args:
        indexes = np.where((y == val).all(axis=1))[0]
        percentage_val = math.ceil(len(indexes)*percentage)
        indexes = indexes[0:percentage_val]
        extr_x = x[indexes]
        extr_y = y[indexes]
        r.append((extr_x, extr_y))
        x = np.delete(x, indexes, axis=0)
        y = np.delete(y, indexes, axis=0)
    r.insert(0,y)
    r.insert(0,x)
    return  tuple(r)

from NR.nural_network import RFBN_MODELS_PATH, MLP_MODELS_PATH
from NR.perceptron_keras import MLP
from NR.RFBN import RFBN


def _test_model(type, name , data):
    if type == MLP:
        model = MLP(name=name)
        model.load_models(MLP_MODELS_PATH)
    else:
        model = RFBN(name=name)
        model.load_models(RFBN_MODELS_PATH)
    p = model.predict_with_membership_degree(np.array(data))
    print('%s prediction is %s'%(name, str(p)))


def test_model():
    data = np.array([[15, 49]])
    _test_model(MLP, 'perceptron_ar1.csv_22', data)
    _test_model(RFBN, 'rfbn_ar1.csv_22', data)
    _test_model(MLP, 'perceptron_ar1.csv_17', data)

    _test_model(RFBN, 'rfbn_ar1.csv_16', data)
    _test_model(MLP, 'perceptron_ar1.csv_15', data)
    _test_model(MLP, 'perceptron_ar1.csv_14', data)




from collections import defaultdict
def get_columns(base_dir, files, columns):
    column_data = defaultdict(list)
    for f in files:
        pth = path.join(base_dir,f)
        with open(pth, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            for row in reader:
                for column in columns:
                    column_data[column].append(row[column])
    hdrs=''
    dlen = 0
    for headers in column_data.keys():
        hdrs+=headers+'\t\t'
        dlen = len(column_data[headers])
    print(hdrs,'\n')
    for index in range(dlen):
        row=''
        for h in column_data.keys():
            row+= column_data[h][index]+'\t\t'
        print(row)



import math
if __name__ == '__main__':
  test_model()