import utils.datapartitional as util
from utils.one_hot_encoder import OneHotEncoder
import pandas as pd
from NR.perceptron_keras import MLP
from NR.RFBN import RFBN
from utils import filesystem as fs
from NR.nural_network import RFBN_MODELS_PATH, MLP_MODELS_PATH
from itertools import combinations
import logging

LOGGER = logging.getLogger(__name__)


def create_ANN_models( model_path, use_case, dataset, features_col_names, class_col_name, nn_model_strategy='retrain', perceptron_init_param = None, rfbn_init_param = None ):
    '''
    :run_id: unique run_id used to save model for specific use case
    :param dataset: training dataset
    :param features_col_names: training data features columns names
    :param class_col_name: training data class column name
    :param nn_model_strategy: strategy of creating neural network models
                     'reuse' reuse saved model if exists without training , retrain new one if no model is saved
                     'retrain' discard saved model if exists, train new model
                     'train' if saved model exists , train using its weights as initial values , else train new model
    :param perceptron_init_param:
    :param rfbn_init_param:
    :return:
    '''
    models = []
    ohenc = OneHotEncoder([False, True])

    balanced_sets = util.create_balanced_buckets(dataset,class_col_name)
    perceptron_template = 'perceptron_{}_{}_{}##{}'
    rfbn_template = 'rfbn_{}_{}_{}##{}'
    counter=1
    buckets_count = len(balanced_sets)
    comb_len = sum(1 for ignore in combinations(features_col_names,2))
    LOGGER.info('will train %d models for %d buckets', buckets_count*2*comb_len, buckets_count)
    for balanced_set in balanced_sets:
        LOGGER.debug('balanced dataset shape = %s',balanced_set.shape)
        for f1, f2 in combinations(features_col_names,2):
            perceptron_model_name = perceptron_template.format(use_case, counter, f1, f2)
            rbf_model_name = rfbn_template.format(use_case, counter, f1, f2)

            x = np.asarray(balanced_set[[f1, f2]])
            y = ohenc.encode(balanced_set[class_col_name].tolist())

            mlp = _get_nn_model(x, y, model_path, [f1, f2], perceptron_model_name, MLP, nn_model_strategy, visualise= True, **perceptron_init_param )
            if mlp.is_avg_matrix_gt(0.65, 'val_recall') and mlp.is_avg_matrix_gt(0.65, 'recall'):
                models.append(mlp)

            rfbn = _get_nn_model(x, y, model_path, [f1, f2], rbf_model_name, RFBN, nn_model_strategy, visualise= True, **rfbn_init_param )
            if rfbn.is_avg_matrix_gt(0.65, 'val_recall') and rfbn.is_avg_matrix_gt(0.65, 'recall'):
                models.append(rfbn)

        counter+=1
    return models


def _get_nn_model(x,y, run_path, features_names, model_name, model_type, nn_model_strategy, visualise=False, **kwargs):
    LOGGER.debug("create or load model [%s]", model_name)
    if model_type == MLP:
        model = MLP(model_name, features_names, visualize=visualise, **kwargs)
        path = fs.join(run_path, MLP_MODELS_PATH)
    else:
        model = RFBN(model_name, features_names, visualize=visualise, **kwargs)
        path = fs.join(run_path, RFBN_MODELS_PATH)

    if nn_model_strategy in ['reuse','train']:
        is_loaded = model.load_models(path)
        if is_loaded:
            if nn_model_strategy == 'reuse':
                return model
            else:
                model.train_model(x, y, tensorboard_dir=path)
                model.save(path)
        else:
            fs.delete(path, model_name)
            model.train_model(x, y)
            model.save(path)

    elif nn_model_strategy == 'retrain':
        fs.delete(path, model_name)
        model.train_model(x, y, tensorboard_dir=path)
        model.save(path)

    return model



import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def visualize(rbf,data_set,features_columns,class_column):

    x = data_set[features_columns[0]]
    y = data_set[features_columns[1]]
    c = data_set[class_column]
    centers = rbf.train_history.model.get_layer(name='rbf_layer_name').weights[0].numpy()
    cx = centers[:,0]
    cy = centers[:,1]
    colors = ['red', 'green']

    fig1 = plt.figure(figsize=(4, 4) )
    plt.scatter(x, y, c=c, cmap=matplotlib.colors.ListedColormap(colors))
    plt.scatter(cx, cy , color='b')
    plt.pause(0.05)
    cb = plt.colorbar()
    loc = np.arange(0, max(c), max(c) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)


import sys
if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    ohenc = OneHotEncoder([False, True])
    features_columns = ['unique_operators', 'halstead_vocabulary', 'unique_operands']
    class_column = 'defects'
    data_set =  util.concat_datasets_files(
        ['../test_data/ar1.csv', '../test_data/ar4.csv', '../test_data/ar5.csv', '../test_data/ar6.csv'])

    test_data_set = pd.read_csv('../test_data/ar3.csv', index_col=False)
    test_data_set_x, test_data_set_y = (test_data_set[features_columns], ohenc.encode(test_data_set[class_column]).tolist())

    test_data_set_true = test_data_set[test_data_set[class_column] == True]
    test_data_set_true_x, test_data_set_true_y = (test_data_set_true[features_columns], ohenc.encode(test_data_set_true[class_column].tolist()))

    test_data_set_false = test_data_set[test_data_set[class_column] == False]
    test_data_set_false_x, test_data_set_false_y = (test_data_set_false[features_columns], ohenc.encode(test_data_set_false[class_column].tolist()))

    PERCEPTRON_INIT_PARAM = {
        'learning_rate':0.1,
        'momentum':0.1,
        'decay':0.01,
        'input_shape': (3,),
        'use_bias': True,
        'batch_size': 10,
        'epochs': 400
        ,'loss': 'mean_squared_error'
    }

    RFBN_INIT_PARAM = {
        'betas': 0.5,
        'centers': 2,
        'input_shape': (3,),
        'use_bias': True,
        'batch_size': 50,
        'epochs': 400,
        'loss':'mean_squared_error'
    }

    trained_models = create_ANN_models(data_set,features_columns,class_column, perceptron_init_param=PERCEPTRON_INIT_PARAM, rfbn_init_param=RFBN_INIT_PARAM)

    for tm in trained_models:
        #if isinstance(tm, MLP):
        #    continue
        print('===================== model [{}]=================='.format(tm.identifier))
        overall_score =  tm.score(np.asarray(test_data_set_x), np.asarray(test_data_set_y))
        overall_predict = tm.predict_dataset_with_membership_degree(np.asarray(test_data_set_x))

        #visualize(tm, test_data_set,features_columns,class_column)
        true_class_score = tm.score(np.asarray(test_data_set_true_x), np.asarray(test_data_set_true_y))
        true_class_predict = tm.predict_dataset_with_membership_degree(np.asarray(test_data_set_true_x))

        false_class_score = tm.score(np.asarray(test_data_set_false_x), np.asarray(test_data_set_false_y))
        false_class_predict = tm.predict_dataset_with_membership_degree(np.asarray(test_data_set_false_x))
        print('over all score = ', overall_score)
        print('true class score = ', true_class_score)
        print('false class score = ', false_class_score)
        predicted = tm.predict_dataset_with_membership_degree(np.asarray(test_data_set_x))
        #for index, test_r in enumerate(test_data_set[features_columns + [class_column]].values):
        #    print(test_r , ' -- prediction', predicted[index], '\n' )
    plt.show()