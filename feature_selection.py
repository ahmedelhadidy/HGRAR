import numpy as np
import pandas as pd
from utils import datapartitional as util

def entropy_calculator(sub_set_above):
    class_column = sub_set_above[:,-1]
    entropy = 0
    _,count = np.unique(class_column,  return_counts=True)
    for c in count:
        entropy += -c/class_column.size * np.log2(c/class_column.size)
    return entropy

def entropy_best_split(data, entropy, count = 1):
    size, columns_count = data.shape
    # un consider class column
    columns_count -= 1
    best_splits = []
    result = []
    for c_index in range(columns_count):
        max_info_gain = 0
        max_info_gain_col_val = None
        for c_unique_values in np.unique(data[:, c_index]):
            sub_set_below = data[data[:, c_index] < c_unique_values]
            sub_set_below_size, _ = sub_set_below.shape
            sub_set_above = data[data[:, c_index] >= c_unique_values]
            sub_set_above_size, _ = sub_set_above.shape
            info_gain = entropy - (entropy_calculator(sub_set_above) * sub_set_above_size / size) - (entropy_calculator(sub_set_below) * sub_set_below_size / size)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_col_val = c_unique_values
        best_splits.append(
            {'column_index': c_index, 'best_split_value': max_info_gain_col_val, 'info_gain': max_info_gain})
    while count > 0 :
        count -= 1
        best_split =  max(best_splits, key=lambda x: x['info_gain'])
        result.append(best_split)
        best_splits.remove(best_split)
    return result

def read_dataset(file):
    return pd.read_csv(file, index_col=False)


if __name__ == '__main__' :
    dataset_files= ['ar1.csv','ar3.csv','ar4.csv','ar5.csv','ar6.csv']
    base_dire = 'test_data/'
    datasets = util.concat_datasets_files(dataset_files,base_dire)
    highest_info_gain_features = []
    for dataset in [datasets]:
        entropy = entropy_calculator(dataset.values)
        features = entropy_best_split(dataset.values, entropy, count = 10)
        for fe in features:
            column_name = dataset.columns[fe['column_index']]
            fe['column_name'] = column_name
        highest_info_gain_features.extend(features)
    highest_info_gain_features = sorted(highest_info_gain_features, key=lambda x:x['info_gain'], reverse= True)
    for i in highest_info_gain_features:
        print(i,'\n')