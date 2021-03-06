
from collections import defaultdict
import numpy as np

PREDICT_OBJ_ROW_DATA='row_data'
PREDICT_OBJ_ROW_DATA_COLS='row_data_columns'
PREDICT_OBJ_CLASS='true_class'
PREDICT_OBJ_PREDICTION='prediction'
PREDICT_OBJ_PREDICTION_RESULT='prediction_result'

CONFUSION_MATRIX_TRUE_POSITIVE='TP'
CONFUSION_MATRIX_FALSE_POSITIVE='FP'
CONFUSION_MATRIX_TRUE_NEGATIVE='TN'
CONFUSION_MATRIX_FALSE_NEGATIVE='FN'

CONFUSION_MATRIX_TOT_POSITIVE = 'TOT_POS'
CONFUSION_MATRIX_TOT_NEGATIVE = 'TOT_NEG'
CONFUSION_MATRIX_TOT_PREDICTED_POSITIVE = 'TOT_PRE_POS'
CONFUSION_MATRIX_TOT_PREDICTED_NEGATIVE = 'TOT_PRE_NEG'
CONFUSION_MATRIX_TOT_SAMPLE = 'TOT_SAMPLE'


def create_prediction_obj(**kwargs):
    r"""
    :param kwargs:
      see below
    :keyword Arguments:
     * *row_data* test row
     * *true_class* Actual/True class
     * *prediction* predicted class
    :return:
    object comprise the prediction results
    """
    obj = {}
    obj[PREDICT_OBJ_ROW_DATA] = kwargs[PREDICT_OBJ_ROW_DATA]
    obj[PREDICT_OBJ_ROW_DATA_COLS] = kwargs[PREDICT_OBJ_ROW_DATA_COLS]
    obj[PREDICT_OBJ_CLASS] = kwargs[PREDICT_OBJ_CLASS]
    obj[PREDICT_OBJ_PREDICTION] = kwargs[PREDICT_OBJ_PREDICTION]
    obj[PREDICT_OBJ_PREDICTION_RESULT] = kwargs[PREDICT_OBJ_PREDICTION] == kwargs[PREDICT_OBJ_CLASS]
    return obj

def get_prediction_data(predictions,  *data_col, correct_prediction = True ):
    filter_obj = filter(lambda x:x[PREDICT_OBJ_PREDICTION_RESULT] == correct_prediction, predictions)
    filtered_predictions = list(filter_obj)
    data_array= np.zeros(shape=( len(filtered_predictions), len(data_col) ))
    class_array = np.empty(shape=(len(filtered_predictions),) , dtype=bool)
    for index, pred in enumerate(filtered_predictions):
        all_columns = pred.get(PREDICT_OBJ_ROW_DATA_COLS)
        all_values = pred.get(PREDICT_OBJ_ROW_DATA)
        y= pred.get(PREDICT_OBJ_CLASS)
        req_col_indexes = np.asarray(list([index for index,val in enumerate(all_columns) if val in data_col ]))
        data_array[index] = all_values[req_col_indexes]
        class_array[index] = y

    return data_array, class_array



class Matrix:

    def __init__(self):
        self.__matrix = defaultdict(int)

    def update_matrix_bulk(self, predictions):
        for p in predictions:
            self.update_matrix(p)

    def update_matrix(self, prediction:{}):
        if PREDICT_OBJ_CLASS not in prediction or PREDICT_OBJ_PREDICTION not in prediction:
            raise Exception('prediction object does not contain essential keys {} '.format([PREDICT_OBJ_CLASS, PREDICT_OBJ_PREDICTION]))

        if prediction[PREDICT_OBJ_CLASS] == False and prediction[PREDICT_OBJ_PREDICTION] == False:
            self.__matrix[CONFUSION_MATRIX_TRUE_NEGATIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_NEGATIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_PREDICTED_NEGATIVE] += 1
        if prediction[PREDICT_OBJ_CLASS] == False and prediction[PREDICT_OBJ_PREDICTION] == True:
            self.__matrix[CONFUSION_MATRIX_FALSE_POSITIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_PREDICTED_POSITIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_NEGATIVE] += 1
        if prediction[PREDICT_OBJ_CLASS] == True and prediction[PREDICT_OBJ_PREDICTION] == False:
            self.__matrix[CONFUSION_MATRIX_FALSE_NEGATIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_POSITIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_PREDICTED_NEGATIVE] += 1
        if prediction[PREDICT_OBJ_CLASS] == True and prediction[PREDICT_OBJ_PREDICTION] == True:
            self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_POSITIVE] += 1
            self.__matrix[CONFUSION_MATRIX_TOT_PREDICTED_POSITIVE] += 1
        self.__matrix[CONFUSION_MATRIX_TOT_SAMPLE] += 1

    def accuracy(self):
        try:
            return (self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE] + self.__matrix[CONFUSION_MATRIX_TRUE_NEGATIVE])\
                   / self.__matrix[CONFUSION_MATRIX_TOT_SAMPLE]
        except ZeroDivisionError:
            return None

    def precision(self):
        try:
            true_pos = self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE]
            false_pos = self.__matrix[CONFUSION_MATRIX_FALSE_POSITIVE]
            return  true_pos/(true_pos+false_pos)
        except ZeroDivisionError:
            return None

    def specificity(self):
        try:
            return self.__matrix[CONFUSION_MATRIX_TRUE_NEGATIVE] / \
                   (self.__matrix[CONFUSION_MATRIX_TRUE_NEGATIVE] + self.__matrix[CONFUSION_MATRIX_FALSE_POSITIVE])
        except ZeroDivisionError:
            return None

    def recall(self):
        try:
            return self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE] / \
                   (self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE] + self.__matrix[CONFUSION_MATRIX_FALSE_NEGATIVE])
        except ZeroDivisionError:
            return None

    def score(self):
        return self.accuracy()

    def AUC(self):
        return (self.recall() + self.specificity()) / 2

    def get_TP( self ):
        return self.__matrix.get(CONFUSION_MATRIX_TRUE_POSITIVE)

    def get_FP( self ):
        return self.__matrix.get(CONFUSION_MATRIX_FALSE_POSITIVE)

    def get_TN( self ):
        return self.__matrix.get(CONFUSION_MATRIX_TRUE_NEGATIVE)

    def get_FN( selsf ):
        return selsf.__matrix.get(CONFUSION_MATRIX_FALSE_NEGATIVE)

    def __str__(self):
        return str(self.__matrix)

