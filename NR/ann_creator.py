import utils.datapartitional as util
from utils.one_hot_encoder import OneHotEncoder
import pandas as pd
from NR.perceptron_keras import MLP
from NR.RFBN import RFBN
from utils import filesystem as fs
from NR.nural_network import RFBN_MODELS_PATH, MLP_MODELS_PATH
from itertools import combinations
import logging
import numpy as np
import sys
LOGGER = logging.getLogger(__name__)


def create_ANN_models( model_path, use_case, dataset, features_col_names, class_col_name,use_nn_types = ('ALL'), nn_model_strategy='retrain', balanced_buckets=True,
                       merge_buckets = False, perceptron_init_param = None, rfbn_init_param = None ):
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
    balanced_sets = _buckets(dataset, class_col_name, balanced_buckets=balanced_buckets, merge_buckets=merge_buckets)
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
            if 'ALL' in use_nn_types or MLP.__name__ in use_nn_types:
                mlp = _get_nn_model(x, y, model_path, [f1, f2], perceptron_model_name, MLP, nn_model_strategy, visualise= True, **perceptron_init_param )
                #if mlp.is_metrics_gt(('recall',0.65), ('val_recall',0.6 ), ('precision',0.65 ), ('val_precision', 0.6)) :
                models.append(mlp)
            if 'ALL' in use_nn_types or RFBN.__name__ in use_nn_types:
                rfbn = _get_nn_model(x, y, model_path, [f1, f2], rbf_model_name, RFBN, nn_model_strategy, visualise= True, **rfbn_init_param )
                #if rfbn.is_metrics_gt(('recall',0.65), ('val_recall',0.6 ), ('precision',0.65 ), ('val_precision', 0.6)):
                models.append(rfbn)

        counter+=1
    return models

def extend_ANN_models_training( model_path, dataset, features_col_names, class_col_name,  balanced_buckets=True ):
    '''
    extend nn models training on new data
    :param model_path:
    :param dataset:
    :param features_col_names:
    :param class_col_name:
    :param balanced_buckets:
    :param merge_buckets:
    :return:
    '''
    ohenc = OneHotEncoder([False, True])
    balanced_set = _buckets(dataset, class_col_name, balanced_buckets=balanced_buckets, merge_buckets=True)[0]
    saved_models_dirs = fs.get_all_directories_contains(model_path , 'saved_model.pb')
    loaded_models = []
    for saved_models_dir in saved_models_dirs:
        remains, model_name = fs.split(saved_models_dir)
        _, model_type_dir = fs.split(remains)
        if MLP.__name__.lower()+'_models' == model_type_dir:
            model_type = MLP
        elif RFBN.__name__.lower()+'_models' == model_type_dir:
            model_type = RFBN
        else:
            raise Exception('model saved sub dir %s does not match any model type '% model_type_dir)
        m = _get_nn_model(None, None, features_col_names, model_name, model_type,'just_load')
        if m:
            loaded_models.append(m)
    LOGGER.debug('loaded : %d models for retraining ', len(loaded_models))
    for f1, f2 in combinations(features_col_names, 2):
        LOGGER.debug('retrain models on features : [ %s , %s ]', f1, f2)
        x = np.asarray(balanced_set[[f1, f2]])
        y = ohenc.encode(balanced_set[class_col_name].tolist())
        for model in loaded_models:
            if model.is_trained_on(f1, f2):
                LOGGER.debug('retraining model : %s on new data', model.identifier)
                model.train_model(x, y, build_model=False)


def _get_nn_model(x,y, run_path, features_names, model_name, model_type, nn_model_strategy, visualise=False, **kwargs):
    LOGGER.debug("create or load model [%s]", model_name)
    if model_type == MLP:
        model = MLP(model_name, features_names, visualize=visualise, **kwargs)
        path = fs.join(run_path, MLP_MODELS_PATH)
    else:
        model = RFBN(model_name, features_names, visualize=visualise, **kwargs)
        path = fs.join(run_path, RFBN_MODELS_PATH)

    if nn_model_strategy in ['reuse','train', 'just_load']:
        is_loaded = model.load_models(path)
        if is_loaded:
            if nn_model_strategy in ('reuse','just_load'):
                return model
            else:
                model.train_model(x, y, build_model=False, tensorboard_dir=path)
                model.save(path)
        else:
            if nn_model_strategy == 'just_load':
                return None
            fs.delete(path, model_name)
            model.train_model(x, y, tensorboard_dir=path)
            model.save(path)

    elif nn_model_strategy == 'retrain':
        fs.delete(path, model_name)
        model.train_model(x, y, tensorboard_dir=path)
        model.save(path)

    return model


def _buckets(dataset, class_col_name,balanced_buckets= True, merge_buckets= False):
    if balanced_buckets:
        balanced_sets = util.create_balanced_buckets(dataset,class_col_name)
        if merge_buckets:
            balanced_sets = [util.concat_datasets(balanced_sets)]
    else:
        balanced_sets=[dataset]
    return balanced_sets


