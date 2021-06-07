
from NR.RBF_NN.rbflayer import RBFLayer, InitCentersRandom
from NR.RBF_NN.kmeans_initializer import InitCentersKMeans
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from NR.nural_network import Basic_NN, get_normalizer_layer, visualize
import tensorflow.keras.metrics as m
from tensorflow.keras.callbacks import EarlyStopping


class RFBN(Basic_NN):
    RBF_LAYER_NAME='rbf_layer_name'
    train_history=None

    def _build_model( self, x, y):
        params = self.model_params
        normalization_layer = get_normalizer_layer(params.get('input_shape', (2,)),x)
        initializer =  kmean_initializer(normalization_layer(x).numpy())
        opt_rmsprop = RMSprop(learning_rate=0.1, momentum=0.1, decay=0.1)
        opt_adam = Adam(learning_rate=0.05)
        model = Sequential([
            normalization_layer,
            RBFLayer(params.pop('centers', 2), alpha=0.5, initializer=initializer,name=self.RBF_LAYER_NAME),
            #Dense(15, activation=activations.sigmoid),
            #Dense(15, activation=activations.sigmoid),
            Dense(2,  activation=activations.softmax)
        ])
        model.compile(loss=params.get('loss','binary_crossentropy'), optimizer=opt_rmsprop,
                      metrics=['accuracy',m.Recall(), m.Precision()])
        model.summary()
        print('before training ', model.get_layer(self.RBF_LAYER_NAME).get_weights())
        return model

    def _train_model( self, x, y, model ):
        params = self.model_params
        epochs = params.get('epochs', 2000)
        patience_ration = params.get('early_stop_patience_ratio', 0.1)
        stop_monitor_metrics = params.get('early_stop_monitor_metric', 'loss')
        patience = int(epochs * patience_ration)
        early_stop_callback = EarlyStopping(monitor=stop_monitor_metrics, patience=patience, verbose=2,
                                                               restore_best_weights=True)
        self.train_history = model.fit(x, y, batch_size=params.get('batch_size',10),epochs=epochs,validation_split=0.2,
                                       callbacks=[early_stop_callback])
        print('after training ', model.get_layer(self.RBF_LAYER_NAME).get_weights())
        if self.visualize:
            visualize(self.train_history)


def random_initializer( x ):
    return InitCentersRandom(x)


def kmean_initializer( x ):
    return InitCentersKMeans(x, max_iter=500)


