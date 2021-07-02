
from NR.RBF_NN.rbflayer import RBFLayer, InitCentersRandom
from NR.RBF_NN.kmeans_initializer import InitCentersKMeans
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from NR.nural_network import Basic_NN, get_normalizer_layer
import tensorflow.keras.metrics as m
from tensorflow.keras.callbacks import EarlyStopping
import utils.filesystem as fs
from tensorflow.keras import regularizers


class RFBN(Basic_NN):
    RBF_LAYER_NAME='rbf_layer_name'

    def _build_model( self, x, y):
        params = self.model_params
        normalization_layer = get_normalizer_layer(params.get('input_shape', (2,)),x)
        #initializer =  kmean_initializer(normalization_layer(x).numpy())
        initializer = kmean_initializer(x)
        opt_rmsprop = RMSprop(learning_rate=params.get('learning_rate',0.1), momentum=params.get('momentum',0.0),
                              decay=params.get('decay',0.1))
        opt_adam = Adam(learning_rate=0.05)
        model = Sequential([
           # normalization_layer,
            Input(shape=params.get('input_shape', (2,))),
            RBFLayer(params.pop('centers', 2), alpha=params.get('alfa', 0.5) , p=params.get('p', 1),
                     initializer=initializer,name=self.RBF_LAYER_NAME),
            Dense(2,  activation=activations.softmax)
        ])
        model.compile(loss=params.get('loss','binary_crossentropy'), optimizer=opt_rmsprop,
                      metrics=['accuracy',m.Recall(name='recall'), m.Precision(name='precision')], run_eagerly=False)

        model.summary()

        return model

    def _train_model( self, x, y, x_val, y_val, model ):

        before = model.get_layer(self.RBF_LAYER_NAME).get_weights()[0]
        params = self.model_params
        epochs = params.get('epochs', 2000)
        patience_ration = params.get('early_stop_patience_ratio', 0.1)
        stop_monitor_metrics = params.get('early_stop_monitor_metric', 'loss')
        if patience_ration <= 1 :
            patience = int(epochs * patience_ration)
        else:
            patience = patience_ration
        early_stop_callback = EarlyStopping(monitor=stop_monitor_metrics, patience=patience, verbose=2,
                                                               restore_best_weights=True)

        train_history = model.fit(x, y,epochs=epochs,batch_size=params.get('batch_size',10),validation_data=(x_val, y_val),
                                       callbacks=[early_stop_callback])
        after = model.get_layer(self.RBF_LAYER_NAME).get_weights()[0]
        print('before training ', before, '\n', 'after training ', after)
        return train_history


def random_initializer( x ):
    return InitCentersRandom(x)


def kmean_initializer( x ):
    return InitCentersKMeans(x, max_iter=500)


