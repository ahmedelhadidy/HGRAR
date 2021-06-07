from NR.nural_network import Basic_NN, visualize, get_normalizer_layer
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations, optimizers
import tensorflow.keras.metrics as m
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

class MLP(Basic_NN):

    def _build_model( self,x,y):

        params = self.model_params
        normalization_layer = get_normalizer_layer(params.get('input_shape', (2,)), x)
        model = Sequential([
            normalization_layer,
            #tf.keras.layers.Dense(20, activation=activations.sigmoid),
            tf.keras.layers.Dense(2, activation=activations.sigmoid),
            tf.keras.layers.Dense(2, activation=activations.softmax)
        ])

        opt= optimizers.RMSprop(learning_rate=params.get('learning_rate',0.1), momentum=params.get('momentum',0.1),
                                decay=params.get('decay',0.1))
        model.compile(loss=params.get('loss', 'binary_crossentropy'), optimizer=opt,  metrics=['accuracy', m.Precision(), m.Recall()])
        model.summary()
        return model

    def _train_model( self, x, y, model ):
        params = self.model_params
        epochs = params.get('epochs',2000)
        patience_ration = params.get('early_stop_patience_ratio', 0.1)
        stop_monitor_metrics = params.get('early_stop_monitor_metric', 'loss')
        patience = int(epochs * patience_ration)
        early_stop_callback = EarlyStopping(monitor=stop_monitor_metrics, patience=patience,verbose=2, restore_best_weights=True)
        self.train_history = model.fit(x, y, batch_size=params.get('batch_size',10),epochs=epochs,validation_split=0.2,
                                       callbacks=[early_stop_callback])
        if self.visualize:
            visualize(self.train_history)
