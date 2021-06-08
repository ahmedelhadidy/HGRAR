import os
import os.path as path
import tensorflow as tf
from utils.timer import Timer
from tensorflow.keras.layers.experimental.preprocessing import Normalization

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
        self._train_model(x,y, self._model)

    def _train_model( self,x,y, model):
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
def visualize(history):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15, 8))
    fig.tight_layout(pad=4.0)
    ax1.set_title('Training VS Validation loss')
    plt.subplots_adjust(top=0.8)
    color = 'tab:green'
    color_val = 'tab:blue'

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax1.plot(history.epoch, history.history['loss'], color=color, label='Training')
    ax1.plot(history.epoch,history.history['val_loss'], color=color_val, label='Validation')

    ax2.set_title('Training VS Validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.plot(history.epoch, history.history['accuracy'], color=color, label='Training')
    ax2.plot(history.epoch,history.history['val_accuracy'], color=color_val, label='Validation')

    plt.legend()
    plt.show()