import logging
import tensorflow as tf
from NR.RFBN import RBFLayer
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import os.path as path
import csv
from tensorflow.keras.initializers import Initializer
from model.grar.operator import Operator
from model.grar.perceptron_operator import AnnOperator
from model.grar.operator import OperatorType
from model.grar.item import Item
from model.grar.term import Term
from model.grar.gRule import GRule
import model.grar.gRule as gRule
from hygrar import HyGRAR
import utils.datapartitional as dutil
import utils.filesystem as fm
from utils.matrix import Matrix
from experiment import print_matrix
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from utils.one_hot_encoder import OneHotEncoder

from NR.RBF_NN.kmeans_initializer import InitCentersKMeans2

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
LOGGER = logging.getLogger(__name__)
logging.disable(logging.NOTSET)

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
    input_data = dutil.concat_datasets_files(['ar1.csv'], base_dire='test_data')
    x = np.asarray(input_data[['unique_operators', 'halstead_vocabulary']])
    y= np.asarray(input_data[['defects']])
    normalizer = Normalization(input_shape=(2,))
    normalizer.adapt(x)
    input_data = normalizer(x)
    f = np.where(y)[0]
    nf = np.where(y == False)[0]
    x_faulty = x[f]
    x_not_faulty = x[nf]

    #initializer=kmean_initializer(x)
    initializer = InitCentersKMeans2(x_faulty,x_not_faulty)
    centers = initializer(shape=(2,2))
    rbf_layer = RBFLayer(2, initializer=initializer, alpha=0.5)
    distances = rbf_layer(tf.constant(x, dtype=float)).numpy()
    print(centers)
    visualize(centers,x,y,distances)


def test_rbf_model():
    run_id='144'
    usecase = 'ar1.csv'
    input_data = dutil.concat_datasets_files([usecase], base_dire='test_data')
    x = np.asarray(input_data[['unique_operators', 'halstead_vocabulary']])
    y= np.asarray(input_data[['defects']])
    normalizer = Normalization(input_shape=(2,), dtype=float)
    normalizer.adapt(x)
    x = normalizer(x)
    f = np.where(y)[0]
    nf = np.where(y == False)[0]


    model = RFBN('rfbn_ar1.csv_2_unique_operators##halstead_vocabulary')
    base = fm.get_relative_to_home('hygrar')
    path = fm.join(base,run_id, usecase, RFBN_MODELS_PATH)
    model.load_models(path)
    rbf_layer = model._model.get_layer(RFBN.RBF_LAYER_NAME)
    centers = rbf_layer.get_weights()[0]

    distances = rbf_layer(x)
    print(centers)
    visualize(centers,x.numpy(),y,distances.numpy())


def visualize(centers, data, y,distances):
    jitter = .3
    f = plt.figure(1, figsize=(15,8))
    c1_dis  = distances[:,0]
    c2_dis = distances[:,1]
    center_dis_arr = np.where(c1_dis < c2_dis, 0,1)
    color_palette={0: 'red', 1:'green'}
    f = np.where(y)[0]
    nf = np.where(y == False)[0]
    data_faulty=data[f]
    data_not_faulty = data[nf]
    center_dis_faulty = center_dis_arr[f]
    print(center_dis_faulty)
    center_dis_non_faulty = center_dis_arr[nf]


    seaborn.scatterplot(x=data_not_faulty[:, 0], y=data_not_faulty[:, 1], hue=center_dis_non_faulty, palette=color_palette,
                        marker='o', x_jitter=-jitter, y_jitter=-jitter)
    seaborn.scatterplot(x=data_faulty[:,0], y=data_faulty[:,1] , hue=center_dis_faulty, palette=color_palette, marker='x' , x_jitter=jitter, y_jitter=jitter)

    seaborn.scatterplot(x=centers[:, 0], y=centers[:, 1], hue=[0,1], palette=color_palette, marker='^')


    #plt.scatter(data_faulty[:,0], data_faulty[:,1],c = center_dis_faulty,cmap=cmaps, marker='x')
    #plt.scatter(data_not_faulty[:, 0], data_not_faulty[:, 1],c=center_dis_non_faulty,cmap=cmaps, marker='o')
    #plt.scatter(centers[:, 0], centers[:, 1],c=[0,1], cmap=cmaps,marker='^')
    plt.show()


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


def _test_model(run_id, case_id, type, name , data):
    base_path = fm.get_relative_to_home('hygrar')
    base_path = fm.join(base_path,run_id,case_id)
    if type == MLP:
        model = MLP(name=name)
        base_path = fm.join(base_path, MLP_MODELS_PATH)
        model.load_models(base_path)
    else:
        model = RFBN(name=name)
        base_path = fm.join(base_path, RFBN_MODELS_PATH)
        model.load_models(base_path)
    p = model.predict_with_membership_degree(data)
    print('%s prediction is %s'%(name, str(p)))


def test_model(run_id, case_id, type, model_name, data):
    _test_model(run_id, case_id, type, model_name, data)





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

def fun(*names , test='dodo'):
    print(len(names))
    for ind, v in enumerate(names):
        print(ind,"  ",v)
    print(test)

def test_op_type():
    term1 = Term(Item('i1',0), AnnOperator(OperatorType.FAULTY, None))
    term2 = Term(Item('i1',1), Operator(OperatorType.NE))
    assert term1.grar_eq(term2)

def test_grul_r1_join():
    term1 = Term(Item('unique_operators',0),
                 AnnOperator(OperatorType.FAULTY, MLP('perceptron_ar1.csv_1_unique_operators##halstead_vocabulary',['unique_operators','halstead_vocabulary'])))
    term2 = Term(Item('halstead_vocabulary',0),Operator(OperatorType.NE))
    term3 = Term(Item('halstead_vocabulary',0),
                 AnnOperator(OperatorType.FAULTY, RFBN('rfbn_ar1.csv_2_halstead_vocabulary##unique_operands',['halstead_vocabulary','unique_operands'])))
    term4= Term(Item('unique_operands',0), Operator(OperatorType.NE))

    rule1 = GRule(2)
    rule1.add_terms([term1, term2])

    rule2 = GRule(2)
    rule2.add_terms([term3, term4])

    a,b,c = rule1.join_r1_detected(rule2)

    print(str(a[0]),str(b[0]),str(c))

def test_grul_terms_identical():
    term1 = Term(Item('unique_operators',0),
                 AnnOperator(OperatorType.FAULTY, MLP('perceptron_ar1.csv_1_unique_operators##halstead_vocabulary',['unique_operators','halstead_vocabulary'])))
    term2 = Term(Item('halstead_vocabulary',0),Operator(OperatorType.NE))
    term3 = Term(Item('halstead_vocabulary',0),
                 AnnOperator(OperatorType.FAULTY, RFBN('rfbn_ar1.csv_2_halstead_vocabulary##unique_operands',['halstead_vocabulary','unique_operands'])))
    term4= Term(Item('unique_operators',0), Operator(OperatorType.NE))

    rule1 = GRule(2)
    rule1.add_terms([term1, term2])

    rule2 = GRule(2)
    rule2.add_terms([term3, term4])

    assert  gRule._is_terms_identical(rule1, rule2)

def get_normalizer(shape):
    datasources_conc = dutil.concat_datasets_files(['ar3.csv', 'ar4.csv', 'ar5.csv', 'ar6.csv'], base_dire='test_data')
    datasources_conc = datasources_conc[['unique_operators', 'halstead_vocabulary', 'unique_operands']]
    datasources_conc_arr = np.asarray(datasources_conc)
    normalizer = Normalization(input_shape=shape, mean=0, variance=1)
    normalizer.adapt(datasources_conc_arr)
    return normalizer

def test_normalization():
    datasources = ['ar1.csv','ar3.csv','ar4.csv','ar5.csv','ar6.csv']
    rows_list = []
    rows_list.append(['unique_operators', 'halstead_vocabulary', 'unique_operands', 'unique_operators_normalized',
                      'halstead_vocabulary_normalized', 'unique_operands_normalized', 'data_set'])
    for ds_name in datasources:
        ds = dutil.concat_datasets_files([ds_name], base_dire='test_data')
        ds = ds[['unique_operators', 'halstead_vocabulary', 'unique_operands']]
        ds_arr = np.asarray(ds)
        len = ds_arr.shape[0]
        input_shape = (3,)
        ds_arr_norm = get_normalizer(input_shape)(ds_arr)
        for i in range(len):
            rows_list.append(np.concatenate((ds_arr[i],ds_arr_norm[i],np.array([ds_name]))))
    with open(fm.get_relative_to_home('normalization_result_mean_0.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_list)

def test_euclidean_distance1():
    a = tf.constant(np.array([[1,2], [3.3, 4.4]]))
    b = tf.constant(np.array([[5.5,6.6],[7,8],[9.9, 10.1]]))
    d = euclidean_distance(a,b)
    tf.print(d)


def test_euclidean_distance2():
    a = tf.constant(np.array([[1, 2, 3,4 ], [-4, -5 ,-6, -7], [6, 7, 8,9]]), dtype=float)
    b = tf.constant(np.array([[1, 2, 3, 4], [1, 1, 1, 1]]), dtype=float)
    d = euclidean_distance(a, b)
    tf.print(d)

def euclidean_distance(input_data,centers):
    na = tf.reduce_sum(tf.square(input_data), 1)
    nb = tf.reduce_sum(tf.square(centers), 1)
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    r = tf.sqrt(tf.maximum(na - 2 * tf.matmul(input_data, centers, False, True) + nb, 0.0))
    return r

def test_saved_hgrar():
    grars_count = 3
    run_id='149'
    ds = dutil.concat_datasets_files(['ar1.csv'], base_dire='test_data')
    hgrar = HyGRAR.load_hgrar(run_id,'ar1.csv', grars_count)
    predictions = hgrar.predict(ds, grars_count)
    matrix = Matrix()
    matrix.update_matrix_bulk(predictions)
    print_matrix(matrix, "prediction on " + 'ar1')

def test_multiple_adapt():
    ar1 = np.array([1,2,3])
    ar2 = np.array([5,6])
    n = Normalization()
    n.adapt(ar1)
    print(n.mean, n.variance , sep='\t')
    n.adapt(ar2, reset_state=False)
    print(n.mean, n.variance, sep='\t')

from NR.normalizers import Normalizer

def test_np_man_min():
    x = np.asarray([[1,2,3], [1,2,3], [4,5,6], [6,5,2]])
    mx = np.max(x, axis=0)
    mi = np.min(x, axis=0)
    minus = mx-mi
    normalizer = Normalizer(0.8)
    normalized = normalizer(x)
    print(normalized)

import math
if __name__ == '__main__':
    #obj = {'unique_operators': 19, 'halstead_vocabulary': 55, 'unique_operands': 36}
    #model_name = 'perceptron_ar1.csv_2_unique_operators##halstead_vocabulary'
    #test_model('149', 'ar1.csv', MLP, model_name, obj)
    #test_saved_hgrar()
    #test_rbf_model()
    test_array=np.array(
        [
            [1,2],
            [4,5],
            [9,10],
            [11,12],
            [22,33]
        ],
        dtype=float
    )

    test_y = np.array([True,False,True,True,False])
    indecies = np.where(test_y)

    test_subt= np.full(fill_value=0.1, shape=(2,), dtype=float)
    print(test_subt)
    test_array[indecies,:] = test_array[indecies,:] - test_subt

    print(test_array)
