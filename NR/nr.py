
import pandas as pd
import utils.datapartitional as dp


class ParentNR:

    def __init__(self, identifier: str, features_col_names:[], class_col_name:str, **kwargs):
        self.classifier= None
        self.identifier = identifier
        self.features_col_names = features_col_names
        self.class_col_name = class_col_name
        self._classifier_param = kwargs

    def train(self,data_set: pd.DataFrame):
        self.classifier = self.init_classifier()
        self.classifier = self.execute_classifier(data_set)
        return self

    def execute_classifier(self, dataset):
        features_data = dataset[self.features_col_names]
        class_data = dataset[self.class_col_name]
        self.classifier = self.classifier.fit(features_data, class_data)
        return self.classifier

    def get_x(self, dataset):
        return dataset[self.features_col_names]

    def get_y(self, dataset):
        return dataset[self.class_col_name]

    def init_classifier(self):
        pass

    def build_prediction_object(self, class_array, prediction):
        res_object = {}
        for indx, val in enumerate(class_array):
            res_object[val] = prediction[indx]
        return res_object

    def predict_with_membership_degree(self, test_data):
        classes = self.classifier.classes_
        predictions = predict_array(self.classifier, test_data)
        _result = []
        for pred in predictions:
            _result.append(self.build_prediction_object(classes,pred))
        return _result

    def predict_dataset_with_membership_degree(self, dataset: pd.DataFrame):
        classes = self.classifier.classes_
        predictions = predict(self.classifier, dataset, self.features_col_names)
        _result = []
        for pred in predictions:
            res_object = {}
            for indx, val in enumerate(classes):
                res_object[val] = pred[indx]
            _result.append(res_object)
        return _result


def _get_balanced_chuncks(data_set: pd.DataFrame , class_column):
    clazz, count = dp.rare_class(data_set, lambda x: x[-1])
    buckets = dp.create_balanced_buckets(data_set, class_column, clazz, count)
    return buckets


def score(model, test_dataset: pd.DataFrame, features: [], class_name: str):
    features_data = test_dataset[features]
    class_data = test_dataset[class_name]
    return model.score(features_data, class_data)


def predict(classifier, dataset: pd.DataFrame, features: []):
    features_data = dataset[features]
    predicted_classes = predict_array(classifier,features_data)
    return predicted_classes


def predict_array(classifier, test_data: []):
    return classifier.predict_proba(test_data)
