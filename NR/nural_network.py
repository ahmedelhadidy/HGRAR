import os
import os.path as path
import tensorflow as tf
from utils.timer import Timer
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import utils.filesystem as fs
import numpy as np
import math

RFBN_MODELS_PATH= fs.get_relative_to_home('rfbn_models')
MLP_MODELS_PATH= fs.get_relative_to_home('mlp_models')

class Basic_NN:

    tf.get_logger().setLevel('ERROR')

    def __init__(self, name, visualize = False , **kwargs):
        self.identifier = name
        self.visualize = visualize
        self.model_params= kwargs

    def _build_model( self, x, y):
        pass

    def train_model( self,x,y ):
        self._model = self._build_model(x,y)
        x, y, (true_validation_x, true_validation_y), (false_validation_x, false_validation_y) = \
            get_value_wise_subsets(x, y, ([0, 1], 0.1), ([1, 0], 0.1))
        x_val = np.concatenate((true_validation_x, false_validation_x), axis=0)
        y_val = np.concatenate((true_validation_y, false_validation_y), axis=0)
        self._train_model(x,y,x_val, y_val, self._model)

    def _train_model( self,x, y, x_val, y_val, model):
        pass

    def load_models( self, model_path ):
        '''
        Load saved Model
        :param model_path: absolute path to the model , not including the model name
        :return: True if model loaded successfully else False
        '''
        try:
            self._model= tf.keras.models.load_model(path.join(model_path, self.identifier))
            return True
        except (IOError, ImportError) as err:
            print(err)
            return False

    def save( self, model_path  ):
        if self._model:
            os.makedirs(path.join(model_path, self.identifier), exist_ok=True)
            self._model.save(path.join(model_path, self.identifier))

    def score( self, x, y ):
        scores = self._model.evaluate(x, y, verbose=0)
        return scores[1]

    @Timer(text="RFBN predict in {:.2f} seconds")
    def predict_with_membership_degree( self, x ):
        predictions = self._model.predict(x)
        class_arr = [False, True]
        result_predictions = []
        for p in predictions:
            result_predictions.append(self.build_prediction_object(class_arr, p))
        return result_predictions

    def predict_dataset_with_membership_degree( self, x ):
        class_arr = [False, True]
        predictions = self._model.predict(x)
        result_predictions = []
        for p in predictions:
            result_predictions.append(self.build_prediction_object(class_arr, p))
        return result_predictions

    def build_prediction_object(self, class_array, prediction):
        res_object = {}
        for indx, val in enumerate(class_array):
            res_object[val] = prediction[indx]
        return res_object


def get_normalizer_layer(input_shape, x):
    normalizer = Normalization(input_shape=input_shape)
    normalizer.adapt(x)
    return normalizer


import matplotlib.pyplot as plt

def next_cell(row, column, rows_count, columns_count):
    if row == rows_count -1 and column == columns_count -1:
        return None, None
    if column == columns_count -1:
        return row+1, 0
    else:
        return row, column+1


def visualize(path, history, *args):
    rows = int(len(args)/2) + len(args) % 2
    cols = 1 if len(args) == 1 else 2
    fig, axes = plt.subplots(rows,cols,figsize=(15, 8))
    axes_shape_length = len(axes.shape)
    fig.tight_layout(pad=4.0)
    plt.subplots_adjust(top=0.8)
    color = 'tab:green'
    color_val = 'tab:red'

    cur_row, cur_col = 0, 0
    for index, matrix in enumerate(args):
        if axes_shape_length == 1:
            ax = axes[index]
        else:
            ax = axes[cur_row][cur_col]
        ax.set_title('Training VS Validation '+matrix)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(matrix)
        ax.plot(history.epoch, history.history[matrix], color=color, label='Training')
        ax.plot(history.epoch,history.history['val_'+matrix], color=color_val, label='Validation')
        cur_row, cur_col = next_cell(cur_row, cur_col, rows, cols)

    plt.legend()
    fs.delete(path)
    plt.savefig(path)


def get_value_wise_subsets(x, y, *args):
    r = []
    for val, percentage in args:
        indexes = np.where((y == val).all(axis=1))[0]
        percentage_val = math.ceil(len(indexes)*percentage)
        indexes = indexes[0:percentage_val]
        extr_x = x[indexes]
        extr_y = y[indexes]
        r.append((extr_x, extr_y))
        x = np.delete(x, indexes, axis=0)
        y = np.delete(y, indexes, axis=0)
    r.insert(0,y)
    r.insert(0,x)
    return  tuple(r)