from utils.matrix import Matrix
from sklearn.model_selection import LeaveOneOut
from datetime import datetime
from utils import datapartitional as util
from hygrar import HyGRAR
import utils.timer as timer
from utils.timer import Timer
import csv
from utils import filesystem as fs

timer.ALLOW_LOGGING=False

timer.ALLOW_LOGGING=False

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
    print('======================={}============================'.format(comment))
    print('score = ', m.score(), 'precision = ', m.precision(), 'Sensitivity/recall = ',
          m.recall(),
          'specificity = ', m.specificity(), 'AUC = ', m.AUC())
    print('matrix : ', m)
    print('===================================================')

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

@Timer(text="one_time_test executed in {:.2f} seconds")
def one_time_test():
    HyGRAR.PERCEPTRON_INIT_PARAM = {
        'learning_rate': 0.1,
        'input_shape': (2,),
        'batch_size': 5,
        'epochs': 800,
        'loss': 'mean_squared_error',
        'early_stop_patience_ratio': 0.7,
        'early_stop_monitor_metric': 'loss',
        'decay': 0.1,
        'momentum': 0.1
    }

    HyGRAR.PFBN_INIT_PARAM = {
        'centers': 2,
        'alfa':0.5,
        'p':1,
        'learning_rate':0.1,
        'decay': 0.1,
        'input_shape': (2,),
        'batch_size': 5,
        'epochs': 800,
        'loss': 'mean_squared_error',
        'early_stop_patience_ratio': 0.7,
        'early_stop_monitor_metric': 'loss'
    }

    hgrar_attributes = {
        'min_s': 1,
        'min_c': 0.5,
        'min_membership': 0.1
    }
    run_id='9'
    all_data_sets, features, class_col, result_prefix, transformer = usecase_data(1)

    results=[]
    for test, train_list in permutations(*all_data_sets):
        print('use case : ', test)
        data_set = util.concat_datasets_files(train_list,base_dire='test_data')
        if transformer:
            data_set = transformer(data_set)
        data_set = data_set[features + [class_col]]
        test_data_set = util.concat_datasets_files([test],base_dire='test_data')[features + [class_col]]
        if transformer:
            test_data_set = transformer(test_data_set)

        hgrar = HyGRAR(test, hgrar_attributes['min_s'], hgrar_attributes['min_c'], hgrar_attributes['min_membership'] ,
                       nn_model_creation='retrain')
        hgrar.train(data_set[features],data_set[class_col])
        predictions  = hgrar.predict(test_data_set,3)
        matrix = Matrix()
        matrix.update_matrix_bulk(predictions)
        results.append((test, matrix))
        create_result_csv(fs.get_relative_to_home(result_prefix+'_results_'+run_id+'.csv'), results)
        print_matrix(matrix, "prediction on "+test)


def usecase_data(id):
    if id == 1:
        all_data_sets = ['ar1.csv', 'ar3.csv', 'ar4.csv', 'ar5.csv', 'ar6.csv']
        features = ['unique_operators', 'halstead_vocabulary']
        class_col = 'defects'
        return  all_data_sets, features, class_col, 'AR', None
    elif id == 2:
        def transform( dataset ):
            faulty = dataset['bug'] > 0
            dataset.drop(columns='bug')
            dataset['bug'] = faulty
            return dataset
        all_data_sets = ['ant-1.7.csv', 'jedit-3.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv','jedit-4.3.csv']
        features = ['wmc', 'cbo']
        class_col = 'bug'
        return all_data_sets, features, class_col, 'jedit_ant', transform


if __name__ == '__main__':
    one_time_test()
