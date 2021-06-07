from utils.matrix import Matrix
from sklearn.model_selection import LeaveOneOut
from datetime import datetime
from utils import datapartitional as util
from hygrar import HyGRAR
from utils.timer import Timer

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

@Timer(text="one_time_test executed in {:.2f} seconds")
def one_time_test():
    features=['unique_operators', 'halstead_vocabulary']
    class_col='defects'
    data_set = util.concat_datasets_files(['ar1.csv','ar4.csv','ar5.csv','ar6.csv'],base_dire='test_data')
    data_set = data_set[features + [class_col]]
    test_data_set = util.concat_datasets_files(['ar3.csv'],base_dire='test_data')[features + [class_col]]
    hgrar_attributes = {
        'min_s': 0.9,
        'min_c': 0.9,
        'min_membership': 0.5
    }
    hgrar = HyGRAR(1, 0.5, 0.0,nn_model_creation='retrain')
    hgrar.train(data_set[features],data_set[class_col])
    predictions  = hgrar.predict(test_data_set,5)
    matrix = Matrix()
    matrix.update_matrix_bulk(predictions)
    print_matrix(matrix, "prediction on ar1")


if __name__ == '__main__':
    one_time_test()
