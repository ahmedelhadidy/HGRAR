import os
import os.path as path
import tensorflow as tf
from utils.timer import Timer
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import utils.filesystem as fs
import numpy as np
import math
import json

RFBN_MODELS_PATH= 'rfbn_models'
MLP_MODELS_PATH= 'mlp_models'

class Basic_NN:

    tf.get_logger().setLevel('ERROR')

    def __init__(self, name,features_names:[], visualize = False , **kwargs):
        self.identifier = name
        self.features_names = features_names
        self.do_visualize = visualize
        self.model_params= kwargs
        self.train_history = None

    def _build_model( self, x, y):
        pass

    def train_model( self,x,y ):
        self._model = self._build_model(x,y)
        x, y, (true_validation_x, true_validation_y), (false_validation_x, false_validation_y) = \
            get_value_wise_subsets(x, y, ([0, 1], 0.1), ([1, 0], 0.1))
        x_val = np.concatenate((true_validation_x, false_validation_x), axis=0)
        y_val = np.concatenate((true_validation_y, false_validation_y), axis=0)
        history =  self._train_model(x,y,x_val, y_val, self._model)
        self.train_history = history.history
        self.train_history['epoch'] = history.epoch
        return self.train_history

    def _train_model( self,x, y, x_val, y_val, model):
        pass

    def load_models( self, model_path ):
        '''
        Load saved Model
        :param model_path: absolute path to the model , not including the model name
        :return: True if model loaded successfully else False
        '''
        try:
            model_path = path.join(model_path, self.identifier)
            self._model= tf.keras.models.load_model(model_path)
            with open(model_path+'.json', 'r') as model_json:
                obj = json.load(model_json)
            self.model_params = obj['model_params']
            self.train_history = obj['training_history']
            self.saved_path = obj['saved_path']
            return True
        except (IOError, ImportError) as err:
            print(err)
            return False

    def save( self, model_path  ):
        if self._model:
            os.makedirs(path.join(model_path, self.identifier), exist_ok=True)
            self._model.save(path.join(model_path, self.identifier))
            self.create_object(model_path)
            if self.do_visualize:
                self.visualize(fs.join(model_path, self.identifier + '.jpg'),
                               'loss', 'accuracy','precision', 'recall')
            self.saved_path = model_path

    def create_object( self, model_path ):
        obj={}
        obj['identifier'] = self.identifier
        obj['class'] = type(self).__name__
        obj['features_names'] = self.features_names
        obj['model_params'] = self.model_params
        obj['training_history'] = self.train_history
        obj['saved_path'] = model_path
        with open(path.join(model_path, self.identifier+'.json'), 'w') as model_file:
            json.dump(obj, model_file, indent=4)


    def score( self, x, y ):
        scores = self._model.evaluate(x, y, verbose=0)
        return scores[1]

    @Timer(text="RFBN predict in {:.2f} seconds")
    def predict_with_membership_degree( self, *x_instances ):
        predictions = self._model.predict(self.x_objects_to_array(*x_instances))
        class_arr = [False, True]
        result_predictions = []
        for p in predictions:
            result_predictions.append(self.build_prediction_object(class_arr, p))
        return result_predictions

    def predict_dataset_with_membership_degree( self, *args ):
        '''
        :param args: one or more duct objects , each one has all features for one test instance
        :return:
        '''
        class_arr = [False, True]
        predictions = self._model.predict(self.x_objects_to_array(*args))
        result_predictions = []
        for p in predictions:
            result_predictions.append(self.build_prediction_object(class_arr, p))
        return result_predictions

    def build_prediction_object(self, class_array, prediction):
        res_object = {}
        for indx, val in enumerate(class_array):
            res_object[val] = prediction[indx]
        return res_object

    def is_trained_on( self, *args ):
        for f_n in args:
            if f_n not in self.features_names:
                return False
        return True

    def x_objects_to_array( self, *x_instances_objects ):
        array = np.empty(shape=(len(x_instances_objects), len(self.features_names)))
        for row_index, obj in enumerate(x_instances_objects):
            for col_index, f_name in enumerate(self.features_names):
                array[row_index][col_index] = obj.get(f_name)
        return array

    def visualize( self, path, *args ):
        rows = int(len(args) / 2) + len(args) % 2
        cols = 1 if len(args) == 1 else 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes_shape_length = len(axes.shape)
        fig.tight_layout(pad=4.0)
        plt.subplots_adjust(top=0.8)
        color = 'tab:green'
        color_val = 'tab:red'
        tick = 50
        epochs_ticks = np.arange(0, self.train_history.get('epoch')[-1] + tick, tick)
        epoch = self.train_history.get('epoch')
        cur_row, cur_col = 0, 0
        for index, matrix in enumerate(args):
            if axes_shape_length == 1:
                ax = axes[index]
            else:
                ax = axes[cur_row][cur_col]
            ax.set_title('Training VS Validation ' + matrix)
            ax.set_xlabel('Epochs')
            ax.set_ylabel(matrix)
            ax.plot(epoch, self.train_history[matrix], color=color, label='Training')
            ax.plot(epoch, self.train_history['val_' + matrix], color=color_val, label='Validation')
            ax.set_xticks(epochs_ticks)
            cur_row, cur_col = next_cell(cur_row, cur_col, rows, cols)

        plt.legend()
        fs.delete(path)
        plt.savefig(path)



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





def get_value_wise_subsets(x, y, *args):
    r = []
    for val, percentage in args:
        indexes = np.where((y == val).all(axis=1))[0]
        if isinstance(percentage, int):
            percentage_val = percentage
        else:
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