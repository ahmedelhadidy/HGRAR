from NR.nural_network import Basic_NN, get_normalizer_layer
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations, optimizers
import tensorflow.keras.metrics as m
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import utils.filesystem as fs
from tensorflow.keras.regularizers import l2 , l1_l2
from tensorflow.keras.layers import BatchNormalization


class MLP(Basic_NN):

    def _build_model( self,x,y):

        params = self.model_params
        lr = params.get('learning_rate', 0.1)
        decay = params.get('decay', 0.1)
        momentum = params.get('momentum', 0.0)
        input_shape = params.get('input_shape', (2,))
        loss_str = params.get('loss', 'binary_crossentropy')
        loss = tf.keras.losses.get(loss_str)
        hidden_layer_nodes = params.get('hidden_neurons', 2)

        #dense_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1)
        dense_initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)

        normalization_layer = get_normalizer_layer(input_shape, x)
        model = Sequential([
            normalization_layer,
            tf.keras.layers.Dense(hidden_layer_nodes, activation=activations.sigmoid ),
            tf.keras.layers.Dense(2, activation=activations.softmax)
        ])
        #, kernel_regularizer='l2', bias_regularizer='l2'
        opt= optimizers.RMSprop(learning_rate=lr, momentum=momentum,decay=decay)
        #opt = optimizers.SGD(learning_rate=lr, momentum=momentum, decay=decay)
        #opt= Adam(learning_rate=params.get('learning_rate', 0.1), decay=params.get('decay',0.1), amsgrad=True)
        #opt = optimizers.SGD(learning_rate=lr, momentum=momentum, decay=decay)

        model.compile(loss=loss,metrics=['accuracy', m.Precision(name='precision'),m.Recall(name='recall')],optimizer=opt, run_eagerly=False)
        model.summary()
        return model

    def _train_model( self, x, y, x_val, y_val, model,callbacks=[] ):
        before = model.get_layer(index=1).get_weights()
        params = self.model_params
        epochs = params.get('epochs',2000)
        batch_size = params.get('batch_size', 10)
        patience_ration = params.get('early_stop_patience_ratio', 0.1)
        stop_monitor_metrics = params.get('early_stop_monitor_metric', 'loss')
        min_delta = params.get('early_stop_min_delta', 0)
        if patience_ration <= 1:
            patience = int(epochs * patience_ration)
        else:
            patience = patience_ration
        early_stop_callback = EarlyStopping(monitor=stop_monitor_metrics, patience=patience,verbose=2, restore_best_weights=True, min_delta=min_delta)
        callbacks.append(early_stop_callback)
        train_history = model.fit(x, y, epochs=epochs,batch_size=batch_size,validation_data =(x_val, y_val), callbacks=callbacks )
        after = model.get_layer(index=1).get_weights()
        print('before training ', before, '\n', 'after training ', after)
        return train_history
