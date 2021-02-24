
from sklearn.neural_network import MLPClassifier
from NR.nr import ParentNR
from utils.timer import Timer
from sklearn import preprocessing

class Perceptron(ParentNR):

    def init_classifier(self):
        if not self._classifier_param:
            return MLPClassifier(activation='logistic', hidden_layer_sizes=(100,100,100), learning_rate='constant', learning_rate_init=0.1
                             ,momentum=0.1,nesterovs_momentum=False ,solver='sgd',max_iter=470, beta_1=0.1,validation_fraction=0.2)
        else:
            return MLPClassifier(**self._classifier_param)

    def transform_y(self, y):
        return  self._labels_encoder.transform(y)

    def _build_labels_encoder(self, lables):
        le = preprocessing.LabelEncoder()
        le.fit(lables)
        return le

    @Timer(text="Perceptron predict in {:.2f} seconds")
    def predict_with_membership_degree(self, test_data):
        return super().predict_with_membership_degree(test_data)

