from  tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


class GradCallback(TensorBoard):
    def __init__(self, model_log_directory, x, y):
        super(GradCallback, self).__init__(log_dir=model_log_directory, histogram_freq=1, write_graph=True, update_freq='epoch' , profile_batch=2, )
        self.var_y_train = tf.Variable(0., validate_shape=False)
        self.var_x_train = tf.Variable(0., validate_shape=False)
        self.x_train = x
        self.y_train = y

    def _log_gradients( self, epoch ):
        step = tf.cast(epoch, dtype=tf.int64)
        writer = self._train_writer
        # writer = self._get_writer(self._train_run_name)

        with writer.as_default(), tf.GradientTape() as g:
            # here we use test data to calculate the gradients
            model : tf.keras.Sequential =self.model
            g.watch(tf.convert_to_tensor(self.x_train))
            _y_pred = model(self.x_train)  # forward-propagation
            loss = model.loss(y_true=self.y_train, y_pred=_y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=step)
        writer.flush()

    def on_epoch_end( self, epoch, logs=None ):
        # def on_train_batch_end(self, batch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super(GradCallback, self).on_epoch_end(epoch, logs=logs)
        # super(ExtendedTensorBoard, self).on_train_batch_end(batch, logs=logs)
        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)
        #self.print_layer(epoch,logs)



