from utils.matrix import Matrix, get_prediction_data
from utils import datapartitional as util
from hygrar import HyGRAR
import utils.timer as timer
from utils.timer import Timer, LOGGER as TIMER_LOGGER
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


def print_matrix(m, comment):
    mertic_str = '======================= %s ============================'
    mertic_str += '\n'
    mertic_str += 'score = %f  precision =  %f  Sensitivity/recall = %f specificity = %f AUC = %f '
    mertic_str += 'metric : %s'
    mertic_str += '\n'
    mertic_str += '==================================================='
    TIMER_LOGGER.info(mertic_str, comment,m.score(), m.precision(), m.recall(), m.specificity(), m.AUC(), str(m))


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
        'learning_rate': 0.005,
        'input_shape': (2,),
        'batch_size': 32,
        'epochs': 5000,
        #'loss': 'mean_squared_error',
        'loss':  'mean_absolute_error',
        #'loss': 'categorical_crossentropy',
        #'loss': 'binary_crossentropy',
        'early_stop_patience_ratio': 0.2,
        'early_stop_monitor_metric': 'val_loss',
        'early_stop_min_delta': 10 ** -3,
        'decay': 0.02,
        'momentum': 0.1
    }

    HyGRAR.PFBN_INIT_PARAM = {
        'centers': 2,
        'alfa': 0.5,
        'p': 1,
        'learning_rate': 0.005,
        'decay': 0.02,
        'momentum': 0.1,
        'input_shape': (2,),
        'batch_size': 32,
        'epochs': 5000,
        #'loss': 'mean_squared_error',
        #'loss': 'categorical_crossentropy',
        'loss': 'mean_absolute_error',
        #'loss': 'binary_crossentropy',
        'early_stop_patience_ratio': 0.2,
        'early_stop_monitor_metric': 'val_loss',
        'early_stop_min_delta': 10 ** -3
    }


@Timer(text="one_time_hygrar_test executed in {:.2f} seconds")
def one_time_hygrar_test(run_id, mode, uscases_names, use_stage_name=None):
    apply_hygrar_nn_parameters()
    hgrar_attributes = {
        'min_s': 1,
        'min_c': 0.5,
        'min_membership': 0.01
    }
    results = defaultdict(list)
    use_nn_types = ('ALL')
    use_cases = get_usecases(uscases_names)

    def create_hygrar():
        return HyGRAR(run_id, name, hgrar_attributes['min_s'], hgrar_attributes['min_c'],
                                   hgrar_attributes['min_membership'],
                                   nn_model_creation='reuse', rule_max_length=rule_length, d1_percentage=0.5)
    for usecase in use_cases:
        uc_name = usecase.get('usecase_name')
        TIMER_LOGGER.info('Test Usecase %s', uc_name)
        test_dateset_names = usecase.get('test_datasets')
        hgrar = None
        for stage in usecase.get('test_stages'):
            stage_name = stage.get('stage_name')
            if use_stage_name and use_stage_name != stage_name:
                continue
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
                hgrar = create_hygrar()
                hgrar.train(training_datasets[features], training_datasets[[class_col]], use_nn_types=use_nn_types, balanced_buckets=True,
                        merge_buckets=True)
            elif mode == 'adaptive_hygrar':
                if not hgrar:
                    hgrar = create_hygrar()
                    hgrar.train(training_datasets[features], training_datasets[[class_col]], use_nn_types=use_nn_types,
                                balanced_buckets=True,
                                merge_buckets=True)
                else: ## extend if exists
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


def test_saved_hgrar(run_id, usecase, grar_count , *visualize_columns , stage=0, bulk = True):
    uc = get_usecases([usecase])[0]
    ds = util.concat_datasets_files(uc.get('test_datasets'), base_dire='test_data')
    name = usecase + '_' + ds.get('test_stages')[stage].get('stage_name')
    if uc.get('dataset_transformation'):
        ds = uc.get('dataset_transformation')(ds)
    hgrar = HyGRAR.load_hgrar(run_id,name, grar_count)
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


def compair(run_id_start):
    hg= 'hygrar_exec'
    ahg= 'adaptive_grar_exec'
    test_cases=['ar1', 'ar3', 'ar4', 'ar5', 'ar6',
                'ant-1.7', 'jedit-3.2', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'
                ]
    #
    with Timer(name=hg):
        one_time_hygrar_test(run_id_start, 'hygrar', test_cases)

    with Timer(name=ahg):
        one_time_hygrar_test(str(int(run_id_start)+1), 'adaptive_hygrar', test_cases)

    TIMER_LOGGER.info('hygrar test executed in : %d seconds , adaptive hygrar test executed in : %d seconds', Timer.timers.get(hg), Timer.timers.get(ahg))


if __name__ == '__main__':
    compair('350')
    #one_time_test('230', 1)
    #one_time_hygrar_test('272', 'hygrar', ['ar1'],  use_stage_name='stage_2')
    #one_time_test('200', 2)
    #one_time_test('141', 2)
    #test_saved_hgrar('215', 'ar1.csv',1, 3, 'unique_operands', 'halstead_vocabulary',bulk=True)
    #test_saved_hgrar('218', 'ar1.csv', 1, 3, 'unique_operands', 'halstead_vocabulary', bulk=True)