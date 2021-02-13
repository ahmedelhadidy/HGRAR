from NR.nr import ParentNR
from NR.RBF_NN.rbflayer import RBFLayer, InitCentersRandom, LabelLimitLayer
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from utils.timer import Timer


class RFBN(ParentNR):
    RBF_LAYER_NAME='rbf_layer_name'

    def init_classifier(self):
        return None

    def __real_init(self,x,**kwargs):
        initializer = InitCentersRandom(x)
        model = Sequential()
        rbflayer = RBFLayer(2,
                            initializer = initializer,
                            betas=kwargs.pop('betas',0.1) ,
                            input_shape=kwargs.pop('input_shape',(2,)), name=self.RBF_LAYER_NAME)
        outputlayer = Dense(2, use_bias=kwargs.pop('use_bias',False))

        model.add(rbflayer)
        model.add(outputlayer)

        model.compile(loss=kwargs.pop('loss','mean_squared_error'), optimizer=RMSprop())
        return model

    def execute_classifier(self, dataset):
        x = self.get_x(dataset).to_numpy()
        y = self.get_y(dataset).to_numpy()
        if self._classifier_param:
            self.classifier = self.__real_init(x, **self._classifier_param)
            self.classifier.fit(x, y, batch_size=self._classifier_param.get('batch_size',10),
                                epochs=self._classifier_param.get('epochs',2000),verbose=0)
        else:
            self.classifier = self.__real_init(x, **{})
            self.classifier.fit(x, y, batch_size=10, epochs=2000, verbose=0)

        return self.classifier

    @Timer(text="RFBN predict in {:.2f} seconds")
    def predict_with_membership_degree(self, test_data):
        predictions = self.classifier.predict(np.asarray(test_data))
        predictions2 = self.classifier.predict_proba(np.asarray(test_data))
        class_arr = [False,True]
        result_predictions = []
        for p in predictions:
            result_predictions.append(self.build_prediction_object(class_arr,p))
        return result_predictions
