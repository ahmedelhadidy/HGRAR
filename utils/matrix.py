
from collections import defaultdict

PREDICT_OBJ_ROW_DATA='row_data'
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
    obj[PREDICT_OBJ_CLASS] = kwargs[PREDICT_OBJ_CLASS]
    obj[PREDICT_OBJ_PREDICTION] = kwargs[PREDICT_OBJ_PREDICTION]
    obj[PREDICT_OBJ_PREDICTION_RESULT] = kwargs[PREDICT_OBJ_PREDICTION] == kwargs[PREDICT_OBJ_CLASS]
    return obj


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
        return (self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE] + self.__matrix[CONFUSION_MATRIX_TRUE_NEGATIVE])\
               / self.__matrix[CONFUSION_MATRIX_TOT_SAMPLE]

    def precision(self):
        return self.__matrix[CONFUSION_MATRIX_TOT_POSITIVE] / self.__matrix[CONFUSION_MATRIX_TOT_PREDICTED_POSITIVE]

    def specificity(self):
        return self.__matrix[CONFUSION_MATRIX_TRUE_NEGATIVE] / \
               (self.__matrix[CONFUSION_MATRIX_TRUE_NEGATIVE] + self.__matrix[CONFUSION_MATRIX_FALSE_POSITIVE])

    def recall(self):
        return self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE] / \
               (self.__matrix[CONFUSION_MATRIX_TRUE_POSITIVE] + self.__matrix[CONFUSION_MATRIX_FALSE_NEGATIVE])

    def score(self):
        return self.accuracy()

    def AUC(self):
        return (self.recall() + self.specificity()) / 2

    def __str__(self):
        return str(self.__matrix)
