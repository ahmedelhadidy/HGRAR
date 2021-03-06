from collections import defaultdict
from pandas import DataFrame
import pandas as pd
import operator
import sys as sys
from itertools import chain

def partition(dataset: DataFrame, partitions_count, class_extractor):
    '''
    Partition dataset to number of partitions denoted by partitions_count , and maintain the proportion of eah class
    in each partition to be same as its proportion in the original dataset
    :param dataset: original data set
    :param partitions_count: number or partitions that original dataset divided to
    :param class_extractor: lambda expression to extract the class from dataset row
    :return:
    '''
    data_length = dataset.shape[0]
    rem = data_length % partitions_count
    partition_length = (data_length - rem) / partitions_count

    classes_proportions_data = defaultdict(list)
    classes_proportions = defaultdict(float)

    for _,row in dataset.iterrows():
        class_val = class_extractor(row)
        classes_proportions_data[class_val].append(row)
        classes_proportions[class_val]+=1

    for clazz in classes_proportions.keys():
        classes_proportions[clazz] = classes_proportions[clazz] / data_length

    partitions = defaultdict(list)

    while __is_data_remaining(classes_proportions_data):
        for i in range(partitions_count):
            partitions[i+1].extend(__create_partition(partition_length, classes_proportions, classes_proportions_data))

    return tuple([pd.DataFrame(v) for v in partitions.values()])


def _rare_class(dataset: pd.DataFrame, class_extractor):
    class_data_length = defaultdict(int)
    for _,row in dataset.iterrows():
        class_data_length[class_extractor(row)] += 1
    sorted_classes = sorted(class_data_length.items(), key=operator.itemgetter(1))
    return sorted_classes[0]


def create_balanced_buckets(dataset: pd.DataFrame, class_column_name):
    '''
    create subsets from dataset, each subset is balanced regarding the number of rows for each class
    :param dataset: dataset
    :param class_column_name: the name of the class column in dataset
    :return: array of datasets where each one has the same rows count that represent each class
    '''
    rare_class_val, rare_class_rows_length = _rare_class(dataset, lambda x:x[class_column_name])
    class_rows = dataset[dataset[class_column_name] == rare_class_val]
    other_class_rows  = dataset[dataset[class_column_name] != rare_class_val]
    other_class_rows_lengh = len(other_class_rows)
    remaining = other_class_rows_lengh % rare_class_rows_length
    buckets_counts = int((other_class_rows_lengh - remaining) / rare_class_rows_length)
    if remaining > 0 or buckets_counts == 0 :
        buckets_counts += 1
    buckets = []
    slice_from_index = 0
    for bucket_index in range(buckets_counts):
        sub = other_class_rows[slice_from_index : slice_from_index + rare_class_rows_length]
        slice_from_index += rare_class_rows_length
        if len(sub) < rare_class_rows_length:
            random_other_class_rows = _random_rows(buckets,class_column_name, not rare_class_val, rare_class_rows_length - len(sub))
            if random_other_class_rows is not None:
                buckets.append(pd.concat([sub, random_other_class_rows, class_rows ], ignore_index = True))
            else:
                buckets.append(pd.concat([sub, class_rows], ignore_index=True))
        else:
            buckets.append(pd.concat([sub, class_rows], ignore_index = True))

    return buckets


def _random_rows(datasets:[], from_column_name ,from_column_value , count: int):
    '''
    Get Random rows from DataFrames datasets denoted by count
    rows taken from datasets equally and the remaining count taken one from each dataset
    :param datasets: DataFrames to get random rows from them
    :param from_column_name: get row based on column name
    :param from_column_value: get rows based on column value
    :param count: count of desigred random rows
    :return: sub DataFrame with rows count equal to count parameter or None if ValueError exception raised
    '''
    if count <= 0 :
        return None
    return_dataset = None
    to_count = 0
    count_from_each_ds = int(count/len(datasets))
    remaining = count % len(datasets)
    try:
        for ds in datasets:
            if return_dataset is not None:
                return_dataset = return_dataset.append(ds[ds[from_column_name] == from_column_value].sample(n=count_from_each_ds, replace=False),
                                                       ignore_index=True)
            else:
                return_dataset = ds[ds[from_column_name] == from_column_value].sample(n=count_from_each_ds, replace=False)
        if remaining > 0:
            while True:
                for ds in datasets:
                    to_count+=1
                    if return_dataset is not None:
                        return_dataset = return_dataset.append(ds[ds[from_column_name] == from_column_value].sample(n=1), ignore_index = True)
                    else:
                        return_dataset = ds[ds[from_column_name] == from_column_value].sample(n=1)
                    if to_count == remaining:
                        return return_dataset
                if to_count == count:
                    return return_dataset
        return return_dataset
    except ValueError:
        print(sys.exc_info())
        return None


def create_uniqe_classes_subsets(dataset: pd.DataFrame, class_column_name):
    '''
    split the given dataset by the value of the column denoted by class_column_name
    :param dataset: dataset
    :param class_column_name: the name of the column where the split carried out based on its values
    :return: dict , where the key is one class value, and the value is sub dataset of rows containing the key as value
             of column denoted by class_column_name
    '''
    uniqe_class_values = set(dataset[class_column_name])
    _return = {}
    for class_val in uniqe_class_values:
        _return[str(class_val)] = dataset[dataset[class_column_name] == class_val]
    return _return


def concat_datasets(datasets: [], axis=0):
    if not is_same_datasets(datasets) and axis == 0:
        raise Exception('datasets are not identical, can''t merge them')
    return pd.concat(datasets, axis=axis)


def concat_datasets_files(datasets_files: [], base_dire=None):
    datasets = []
    base_dire = base_dire[0:-1] if base_dire and base_dire[-1] == '/' else base_dire
    for f in datasets_files:
        absolute_files_path = base_dire+'/'+f if base_dire else f
        datasets.append(pd.read_csv(absolute_files_path,index_col=False))
    if not is_same_datasets(datasets):
        raise Exception('datasets are not identical, can''t merge them')
    return pd.concat(datasets, ignore_index=True)


def is_same_datasets(datasets:[]):
    columns = list(map(lambda x: list(x.columns), datasets))
    data = pd.DataFrame(columns)
    data = data.fillna("")
    data.drop_duplicates(keep="first", inplace=True)
    if len(data) > 1:
        return False
    return True


def get_new_combinations(old_values:[], new_values:[], output_join_length):
    assert output_join_length  > 1
    old_values_participants = output_join_length-1
    i1 = range(len(old_values) + len(new_values))
    i2 = range(len(old_values), len(old_values) + len(new_values))
    all_values = old_values + new_values
    for itr1 in i1:
        for itr2 in i2:
            if itr1 >= itr2:
                continue
            join = [None] * output_join_length
            join[0] = all_values[itr1]
            for remains_participants in range(1, old_values_participants):
                if itr1+remains_participants >= itr2:
                    continue
                join[remains_participants] = all_values[itr1+remains_participants]
            join[-1] = all_values[itr2]
            if None not in join:
                yield tuple(join)


def transform_dataset( dataset ):
    faulty = dataset['bug'] > 0
    dataset.drop(columns='bug')
    dataset['bug'] = faulty
    return dataset

def __create_partition(partition_length, classes_proportions, classes_proportions_data):
    partition_rows = []
    remaining = partition_length
    for clazz in classes_proportions.keys():
        class_data = classes_proportions_data[clazz]
        proportion =  classes_proportions[clazz]
        proportion_size = round(partition_length * proportion )
        if proportion_size <= remaining:
            proportion_sublist = class_data[:proportion_size]
            del class_data[:proportion_size]
            partition_rows.extend(proportion_sublist)
            remaining -= len(proportion_sublist)
        else:
            partition_rows.extend(class_data[:remaining])
            remaining = 0
        if remaining <= 0:
            break
    return partition_rows


def __is_data_remaining(classes_proportions_data: dict):
    for values in classes_proportions_data.values():
        if len(values) > 0 :
            return True
    return False


if __name__ == '__main__':
    for v in get_new_combinations(['a','b'],['E','F','r'], 3):
        print(v)
