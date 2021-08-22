import tensorflow as tf
from NR.RBF_NN.rbflayer import RBFLayer
from NR.RBF_NN.rbflayer_02 import RBFLayer as RBFLayer2
from NR.RBF_NN.kmeans_initializer import InitCentersKMeans, InitCentersKMeans2, InitCentersCMeans
from NR.RBF_NN.initializer import InitCentersRandom
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from NR.nural_network import Basic_NN, get_normalizer_layer
import tensorflow.keras.metrics as m
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
import utils.filesystem as fs
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from utils.one_hot_encoder import OneHotEncoder
from NR.normalizers import Normalizer
import numpy as np


class RFBN(Basic_NN):
    RBF_LAYER_NAME='rbf_layer_name'

    def _build_model( self, x, y):
        params = self.model_params

        input_shape = params.get('input_shape', (2,))
        alpha = params.get('alfa', 0.5)
        p = params.get('p', 1)
        centers = params.get('centers', 2)
        loss_str = params.get('loss', 'binary_crossentropy')
        loss = tf.keras.losses.get(loss_str)
        normalization_layer = get_normalizer_layer(input_shape,x, mean=0, variance=1)
        rbf_init = self.__get_rbf_initializer(normalization_layer(x).numpy())

        #rbf_init = self.__get_rbf_initializer(x)
        #initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1.5)
        #dense_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1)
        opt = self.__get_optimizer()

        model = Sequential([
            normalization_layer,
            #RBFLayer(centers, betas= alpha,name=self.RBF_LAYER_NAME, initializer=rbf_init, dtype=tf.dtypes.double),
            RBFLayer2(centers, alpha=alpha, p=p,  name=self.RBF_LAYER_NAME, initializer=rbf_init, dtype=tf.dtypes.double),
            Dense(2,dtype=tf.dtypes.double,  activation=activations.softmax)
        ])
        model.compile(loss=loss, optimizer=opt,  metrics=['accuracy',m.Recall(name='recall'), m.Precision(name='precision')], run_eagerly=False)

        model.summary()
        return model

    def _train_model( self, x, y, x_val, y_val, model,callbacks=[] ):
        before = model.get_layer(self.RBF_LAYER_NAME).get_weights()
        params = self.model_params
        epochs = params.get('epochs', 2000)
        batch_size = params.get('batch_size',10)
        patience_ration = params.get('early_stop_patience_ratio', 0.1)
        stop_monitor_metrics = params.get('early_stop_monitor_metric', 'loss')
        min_delta = params.get('early_stop_min_delta', 0)
        if patience_ration <= 1 :
            patience = int(epochs * patience_ration)
        else:
            patience = patience_ration
        early_stop_callback = EarlyStopping(monitor=stop_monitor_metrics, patience=patience, verbose=2,min_delta=min_delta,restore_best_weights=True)
        callbacks.append(early_stop_callback)

        terminate_on_nan = TerminateOnNaN()
        callbacks.append(terminate_on_nan)

        train_history = model.fit(x, y,epochs=epochs,batch_size=batch_size , validation_data =(x_val, y_val) , callbacks=callbacks)
        after = model.get_layer(self.RBF_LAYER_NAME).get_weights()
        print('before training ', before, '\n', 'after training ', after)
        return train_history

    def good_to_use( self, accuracy_limit ):
        t_acc = self.train_history.get('accuracy')
        v_acc = self.train_history.get('val_accuracy')
        avg = np.average(np.concatenate((t_acc, v_acc)))
        if avg >= accuracy_limit:
            return True
        return False


    def __get_optimizer( self ):
        params = self.model_params
        lr = params.get('learning_rate', 0.1)
        decay = params.get('decay', 0.1)
        momentum = params.get('momentum', 0.0)
        #opt = RMSprop(learning_rate=lr, momentum=momentum, decay=decay)
        opt = Adam(learning_rate=lr, decay = decay, amsgrad=True)
        #opt = SGD(learning_rate=lr, momentum=momentum, decay=decay)
        #opt = RMSprop()
        return opt

    def __get_rbf_initializer( self,x ):
        #initializer = InitCentersCMeans(x, max_iter=1000)
        # initializer = __rbf_initializer__(normalization_layer(x).numpy(),y)
        initializer = kmean_initializer(x)
        # initializer = random_initializer(normalization_layer(x).numpy())
        #initializer = InitCentersRandom(x)
        return initializer

# def random_initializer( x ):
#     return InitCentersRandom(x)



def kmean_initializer( x ):
    return InitCentersKMeans(x, max_iter=500)

def __rbf_initializer__( x,y ):
    ohe = OneHotEncoder([False, True])
    r_y = np.asarray(ohe.decode(y))
    f = np.where(r_y)[0]
    nf = np.where(r_y == False)[0]
    x_faulty = x[f]
    x_not_faulty = x[nf]
    return InitCentersKMeans2(x_faulty, x_not_faulty, max_iter=500)
