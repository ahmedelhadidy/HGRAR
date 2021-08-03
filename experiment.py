from utils.matrix import Matrix, get_prediction_data
from sklearn.model_selection import LeaveOneOut
from datetime import datetime
from utils import datapartitional as util
from hygrar import HyGRAR
import utils.timer as timer
from utils.timer import Timer
import csv
from utils import filesystem as fs
import logging.config
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from usecases import get_usecases
from collections import defaultdict

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
LOGGER = logging.getLogger(__name__)
logging.disable(logging.NOTSET)
timer.ALLOW_LOGGING=True


def _prepare_data_set(file_names, base_dir, features_col, class_col):
    data_set = util.concat_datasets_files(file_names, base_dir)
    data_set = data_set[features_col + [class_col]]
    return data_set

def llo_cv(dataset, feartures_cols, class_col, **kwargs):
    dataset_x= dataset[feartures_cols]
    dataset_y = dataset[class_col]
    loo = LeaveOneOut()
    matrix_ = Matrix()
    for train_ix, test_ix in loo.split(dataset_x):
        print('LLO CV index out {} - at {}'.format(test_ix,datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        x_train = dataset_x.iloc[train_ix]
        y_train = dataset_y.iloc[train_ix]
        test_dataset = dataset.iloc[test_ix]
        hgrar = HyGRAR(kwargs['min_s'], kwargs['min_c'], kwargs['min_membership'])
        hgrar.train(x_train, y_train)
        predictions = hgrar.predict(test_dataset,5)
        matrix_.update_matrix_bulk(predictions)
    return matrix_

def print_matrix(m, comment):
    mertic_str = '======================= %s ============================'
    mertic_str += '\n'
    mertic_str += 'score = %f  precision =  %f  Sensitivity/recall = %f specificity = %f AUC = %f '
    mertic_str += 'metric : %s'
    mertic_str += '\n'
    mertic_str += '==================================================='
    LOGGER.info(mertic_str, comment,m.score(), m.precision(), m.recall(), m.specificity(), m.AUC(), str(m))


def test_LOO_CV():
    features = ['unique_operators', 'halstead_vocabulary']
    class_col = 'defects'
    data_set_1 = util.concat_datasets_files(['ar1.csv'], base_dire='test_data')
    data_set_3 = util.concat_datasets_files(['ar3.csv'], base_dire='test_data')
    data_set_4 = util.concat_datasets_files(['ar4.csv'], base_dire='test_data')
    data_set_5 = util.concat_datasets_files(['ar5.csv'], base_dire='test_data')
    data_set_6 = util.concat_datasets_files(['ar6.csv'], base_dire='test_data')
    hgrar_attributes = {
        'min_s': 0.9,
        'min_c': 0.9,
        'min_membership': 0.4
    }
    ar1_matrix = llo_cv(data_set_1[features + [class_col]], features, class_col, **hgrar_attributes)
    print_matrix(ar1_matrix, 'ar1')
    ar3_matrix = llo_cv(data_set_3[features + [class_col]], features, class_col, **hgrar_attributes)
    print_matrix(ar3_matrix, 'ar3')
    ar4_matrix = llo_cv(data_set_4[features + [class_col]], features, class_col, **hgrar_attributes)
    print_matrix(ar4_matrix, 'ar4')
    ar5_matrix = llo_cv(data_set_5[features + [class_col]], features, class_col, **hgrar_attributes)
    print_matrix(ar5_matrix, 'ar5')
    ar6_matrix = llo_cv(data_set_6[features + [class_col]], features, class_col, **hgrar_attributes)
    print_matrix(ar6_matrix, 'ar6')


def permutations(*args):
    for item in args:
        yield item, list([i for i in args if i != item])


def create_result_csv(path, results):
    rows_list = [['Case study', 'TP', 'FP', 'TN', 'FN', 'Acc', 'Sens', 'Spec', 'Prec', 'AUC']]
    for test, matrix in results:
        rows_list.append([test, matrix.get_TP(), matrix.get_FP(), matrix.get_TN(), matrix.get_FN(), matrix.accuracy(),
                          matrix.recall(), matrix.specificity(), matrix.precision(), matrix.AUC()])
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_list)

def apply_hygrar_nn_parameters():
    HyGRAR.PERCEPTRON_INIT_PARAM = {
        'hidden_neurons': 2,
        'learning_rate': 0.05,
        'input_shape': (2,),
        'batch_size': 32,
        'epochs': 5000,
        # 'loss': 'mean_squared_error',
        # 'loss':  'mean_absolute_error',
        'loss': 'categorical_crossentropy',
        'early_stop_patience_ratio': 0.1,
        'early_stop_monitor_metric': 'val_loss',
        'early_stop_min_delta': 10 ** -3,
        'decay': 0.1,
        'momentum': 0.1
    }

    HyGRAR.PFBN_INIT_PARAM = {
        'centers': 2,
        'alfa': 0.5,
        'p': 1,
        'learning_rate': 0.1,
        'decay': 0.1,
        'momentum': 0.1,
        'input_shape': (2,),
        'batch_size': 32,
        'epochs': 5000,
        'loss': 'mean_squared_error',
        # 'loss': 'categorical_crossentropy',
        # 'loss': 'mean_absolute_error',
        'early_stop_patience_ratio': 0.1,
        'early_stop_monitor_metric': 'val_loss',
        'early_stop_min_delta': 10 ** -3
    }


@Timer(text="one_time_test executed in {:.2f} seconds", logger=LOGGER.debug)
def one_time_test(run_id, usecase_id):
    apply_hygrar_nn_parameters()
    all_data_sets, features, class_col, result_prefix, transformer = usecase_data(usecase_id)
    hgrar_attributes = {
        'min_s': 1,
        'min_c': 0.5,
        'min_membership': 0.01
    }
    use_nn_types = ('ALL')
    results=[]
    for test, train_list in permutations(*all_data_sets):
        LOGGER.info('use case : %s , train on : %s', test, train_list)
        data_set = util.concat_datasets_files(train_list,base_dire='test_data')
        if transformer:
            data_set = transformer(data_set)
        data_set = data_set[features + [class_col]]
        test_data_set = util.concat_datasets_files([test],base_dire='test_data')[features + [class_col]]
        if transformer:
            test_data_set = transformer(test_data_set)

        hgrar = HyGRAR(run_id,test, hgrar_attributes['min_s'], hgrar_attributes['min_c'], hgrar_attributes['min_membership'] ,
                       nn_model_creation='reuse', rule_max_length=2, d1_percentage=0.7 )
        hgrar.train(data_set[features],data_set[[class_col]], use_nn_types=use_nn_types, balanced_buckets=True, merge_buckets=True)
        hgrar.save_grars()
        predictions  = hgrar.predict(test_data_set,3, bulk=True)
        matrix = Matrix()
        matrix.update_matrix_bulk(predictions)
        results.append((test, matrix))
        create_result_csv(fs.get_relative_to_home(result_prefix+'_results_'+run_id+'.csv'), results)
        print_matrix(matrix, "prediction on "+test)



@Timer(text="one_time_hygrar_test executed in {:.2f} seconds", logger=LOGGER.debug)
def one_time_hygrar_test(run_id, mode, uscases_names):
    apply_hygrar_nn_parameters()
    hgrar_attributes = {
        'min_s': 1,
        'min_c': 0.5,
        'min_membership': 0.01
    }
    results = defaultdict(list)
    use_nn_types = ('ALL')
    use_cases = get_usecases(uscases_names)
    for usecase in use_cases:
        uc_name = usecase.get('usecase_name')
        test_dateset_names = usecase.get('test_datasets')
        hgrar = None
        for stage in usecase.get('test_stages'):
            stage_name = stage.get('stage_name')
            training_datasets_names = stage.get('training_datasets')
            features = stage.get('features')
            rule_length = len(features)
            #predictor_grars_count = 2 if rule_length <= 2 else 3
            predictor_grars_count = 2
            class_col = stage.get('class')
            result_file_prefix = stage.get('result_file_name_prefix')
            transformer = usecase.get('dataset_transformation')

            name = uc_name + '_' + stage_name
            result_file_name = result_file_prefix + '_' + stage_name + '_results_' + run_id + '.csv'

            test_dateset = util.concat_datasets_files(test_dateset_names, base_dire='test_data')
            training_datasets = util.concat_datasets_files(training_datasets_names,base_dire='test_data')
            if transformer:
                test_dateset = transformer(test_dateset)
                training_datasets = transformer(training_datasets)
            if mode == 'hygrar':
                hgrar = HyGRAR(run_id, name, hgrar_attributes['min_s'], hgrar_attributes['min_c'],
                           hgrar_attributes['min_membership'],
                           nn_model_creation='reuse', rule_max_length=rule_length, d1_percentage=0.7)
                hgrar.train(training_datasets[features], training_datasets[[class_col]], use_nn_types=use_nn_types, balanced_buckets=True,
                        merge_buckets=True)
            elif mode == 'adaptive_hygrar':
                if not hgrar:
                    hgrar = HyGRAR(run_id, name, hgrar_attributes['min_s'], hgrar_attributes['min_c'],
                                   hgrar_attributes['min_membership'],
                                   nn_model_creation='reuse', rule_max_length=rule_length, d1_percentage=0.7)
                    hgrar.train(training_datasets[features], training_datasets[[class_col]], use_nn_types=use_nn_types,
                                balanced_buckets=True,
                                merge_buckets=True)
                else:
                    hgrar.extend_training(None, None, training_datasets[features], training_datasets[[class_col]],
                                          extended_rules_max_length=rule_length,use_nn_types=use_nn_types,balanced_buckets=True,
                                          merge_buckets=True,extend_old_grars_training=False)


            hgrar.save_grars()
            predictions = hgrar.predict(test_dateset, predictor_grars_count, bulk=True)
            matrix = Matrix()
            matrix.update_matrix_bulk(predictions)
            results[result_file_name].append((uc_name, matrix))

            create_result_csv(fs.get_relative_to_home(result_file_name), results[result_file_name])
            print_matrix(matrix, "prediction on " + name)




def test_saved_hgrar(run_id, usecase,usecase_id, grar_count , *visualize_columns ,bulk = True):
    _, features, class_col, _, transform = usecase_data(usecase_id)
    ds = util.concat_datasets_files([usecase], base_dire='test_data')
    if transform:
        ds= transform(ds)
    hgrar = HyGRAR.load_hgrar(run_id,usecase, grar_count)
    predictions = hgrar.predict(ds, grar_count, bulk=bulk)
    matrix = Matrix()
    matrix.update_matrix_bulk(predictions)
    print_matrix(matrix, "prediction on " + usecase)
    if len(visualize_columns) == 2:
        corr, corr_y = get_prediction_data(predictions, *visualize_columns)
        incorr, incorr_y = get_prediction_data(predictions, *visualize_columns , correct_prediction=False)
        visualize(corr, corr_y, incorr, incorr_y)


def visualize(corr, corr_y, incorr, incorr_y):
    jitter = .3
    f = plt.figure(1, figsize=(15,8))
    color_palette={'not correct': 'red', 'correct':'green'}
    markers={True:'X', False:'o'}
    corr_pred = np.full(fill_value='correct', shape=(len(corr),))
    incorr_pred = np.full(fill_value='not correct', shape=(len(incorr),))
    pred = np.concatenate((corr_pred, incorr_pred))
    data = np.concatenate((corr, incorr))
    y = np.concatenate((corr_y, incorr_y))

    ytrue = np.where(y)
    data[ytrue,:] = data[ytrue,:] - np.full(fill_value=0.2, shape=(2,), dtype=float)

    seaborn.scatterplot(x=data[:, 0], y=data[:, 1], hue=pred, palette=color_palette,
                        style=y, markers=markers,
                         x_jitter=-jitter, y_jitter=-jitter)

    #seaborn.scatterplot(x=incorr[:, 0], y=incorr[:, 1], hue=np.full(fill_value='not correct', shape=(len(incorr),)),
    #                    palette=color_palette,
    #                    style=np.array(list(['defected' if i else 'not defected' for i in incorr_y])),
    #                    style_order=style_order, x_jitter=jitter, y_jitter=jitter)
    plt.show()


def usecase_data(id):
    if id == 1:
        all_data_sets = ['ar1.csv', 'ar3.csv', 'ar4.csv', 'ar5.csv', 'ar6.csv']
        features = ['unique_operators', 'halstead_vocabulary', 'unique_operands']
        class_col = 'defects'
        return  all_data_sets, features, class_col, 'AR', None
    elif id == 2:
        def transform( dataset ):
            faulty = dataset['bug'] > 0
            dataset.drop(columns='bug')
            dataset['bug'] = faulty
            return dataset
        all_data_sets = ['ant-1.7.csv', 'jedit-3.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv','jedit-4.3.csv']
        features = ['wmc', 'cbo','rfc']
        class_col = 'bug'
        return all_data_sets, features, class_col, 'jedit_ant', transform


if __name__ == '__main__':
    #one_time_test('230', 1)
    one_time_hygrar_test('240', 'adaptive_hygrar', ['ar1', 'ar3', 'ar4', 'ar5', 'ar6'])
    #one_time_test('200', 2)
    #one_time_test('141', 2)
    #test_saved_hgrar('215', 'ar1.csv',1, 3, 'unique_operands', 'halstead_vocabulary',bulk=True)
    #test_saved_hgrar('218', 'ar1.csv', 1, 3, 'unique_operands', 'halstead_vocabulary', bulk=True)