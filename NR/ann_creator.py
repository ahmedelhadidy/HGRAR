import utils.datapartitional as util
import NR.nr as nr
import pandas as pd
from NR.perceptron import Perceptron
from NR.RFBN import RFBN


def create_ANN_models(dataset, features_col_names, class_col_name, perceptron_init_param = None, rfbn_init_param = None):
    models = []
    all_unique_labels=[False,True]
    balanced_sets = util.create_balanced_buckets(dataset,class_col_name)
    perceptron_template = 'perceptron_{}'
    rfbn_template = 'rfbn_{}'
    counter=1
    for balanced_set in balanced_sets:
        models.append(Perceptron(perceptron_template.format(counter), features_col_names, class_col_name,all_unique_labels,**perceptron_init_param).train(balanced_set))
        models.append(RFBN(rfbn_template.format(counter), features_col_names, class_col_name,all_unique_labels, **rfbn_init_param).train(balanced_set))
        counter+=1
    return models


if __name__ == '__main__':
    features_columns = ['unique_operators', 'halstead_vocabulary']
    class_column = 'defects'
    data_set =  util.concat_datasets_files(
        ['../test_data/ar3.csv', '../test_data/ar4.csv', '../test_data/ar5.csv', '../test_data/ar6.csv'])

    test_data_set = pd.read_csv('../test_data/ar1.csv', index_col=False)
    test_data_set_true = test_data_set[test_data_set[class_column] == True]
    test_data_set_false = test_data_set[test_data_set[class_column] == False]
    trained_models = create_ANN_models(data_set,features_columns,class_column)

    for tm in trained_models:
        if not isinstance(tm, Perceptron):
            continue
        print('===================== model [{}]=================='.format(tm.identifier))
        overall_score = nr.score(tm.classifier, test_data_set, features_columns, class_column)
        true_class_score = nr.score(tm.classifier, test_data_set_true,features_columns,class_column)
        false_class_score = nr.score(tm.classifier, test_data_set_false, features_columns, class_column)
        print('over all score = ', overall_score, 'classes =', tm.classifier.classes_)
        print('true class score = ', true_class_score, 'classes =', tm.classifier.classes_)
        print('false class score = ', false_class_score, 'classes =', tm.classifier.classes_)
       # predicted = tm.predict_dataset_with_membership_degree(test_data_set_true)
       # for index, test_r in enumerate(test_data_set_true[features_columns + [class_column]].values):
       #     print(test_r , ' -- prediction', predicted[index], '\n' )