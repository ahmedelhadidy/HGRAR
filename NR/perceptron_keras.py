from NR.nural_network import Basic_NN, get_normalizer_layer
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations, optimizers
import tensorflow.keras.metrics as m
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import utils.filesystem as fs
from tensorflow.keras.regularizers import l2

class MLP(Basic_NN):

    def _build_model( self,x,y):

        params = self.model_params
        normalization_layer = get_normalizer_layer(params.get('input_shape', (2,)), x)

        model = Sequential([
            normalization_layer,
            #tf.keras.layers.Input(shape=params.get('input_shape', (2,))),
            tf.keras.layers.Dense(2, activation=activations.sigmoid),
            tf.keras.layers.Dense(2, activation=activations.softmax)
        ])
        #kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)
        opt= optimizers.RMSprop(learning_rate=params.get('learning_rate',0.1), momentum=params.get('momentum',0.0),decay=params.get('decay',0.1))

        #opt = optimizers.Adam(learning_rate=params.get('learning_rate', 0.1),decay=params.get('decay',0.1))

        model.compile(loss=params.get('loss', 'binary_crossentropy'), optimizer=opt,  metrics=['accuracy', m.Precision(name='precision'),
                                                                                               m.Recall(name='recall')])
        model.summary()
        return model

    def _train_model( self, x, y, x_val, y_val, model ):
        params = self.model_params
        epochs = params.get('epochs',2000)
        patience_ration = params.get('early_stop_patience_ratio', 0.1)
        stop_monitor_metrics = params.get('early_stop_monitor_metric', 'loss')
        if patience_ration <= 1:
            patience = int(epochs * patience_ration)
        else:
            patience = patience_ration
        early_stop_callback = EarlyStopping(monitor=stop_monitor_metrics, patience=patience,verbose=2, restore_best_weights=True)

        train_history = model.fit(x, y, epochs=epochs,batch_size=params.get('batch_size',10),validation_data=(x_val, y_val),
                                       callbacks=[early_stop_callback])

        return train_history
