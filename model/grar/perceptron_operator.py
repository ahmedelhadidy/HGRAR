from model.grar.operator import OperatorType
from model.grar.operator import Operator
from NR.nural_network import Basic_NN
import model.grar.exceptions as exception
from NR.perceptron_keras import MLP
from NR.RFBN import RFBN
import math
import logging

LOGGER = logging.getLogger(__name__)

class AnnOperator(Operator):

    def __init__(self, operator_type : OperatorType, model: Basic_NN):
        if operator_type not in [OperatorType.FAULTY, OperatorType.NON_FAULTY, OperatorType.NE]:
            raise exception.NotSupportedOperator('Operator Type %s Not Supported in class %s'.format(operator_type,type(self)))
        super().__init__(operator_type)
        self.model = model

    def apply(self, *values):
        prediction = self.model.predict_with_membership_degree(*values)[0]
        if self.operator_type == OperatorType.FAULTY:
            member_ship = prediction[True]
        else:
            member_ship = prediction[False]
        if math.isnan(member_ship):
            LOGGER.warning('model %s predictions of values %s is %s', self.model.identifier, values ,prediction)
            return 0
        return member_ship

    # def revers(self):
    #     if self.operator_type == OperatorType.FAULTY:
    #         return AnnOperator(OperatorType.NON_FAULTY, self.model)
    #     else:
    #         return AnnOperator(OperatorType.FAULTY, self.model)

    def revers(self):
        return self

    def create_object( self ):
        obj = {
            "class": "AnnOperator",
            "operator_type": self.operator_type.value,
            "NN_model": {
                "class": type(self.model).__name__,
                "identifier": self.model.identifier,
                "features": self.model.features_names,
                "saved_path": self.model.saved_path
            }
        }
        return obj

    def create_from_obj(  obj ):
        classs = obj.get('NN_model').get('class')
        identifier = obj.get('NN_model').get('identifier')
        features = obj.get('NN_model').get('features')
        saved_path = obj.get('NN_model').get('saved_path')
        if classs == 'MLP':
            model = MLP(identifier, features)
        elif classs == 'RFBN':
            model = RFBN(identifier, features)
        model.load_models(saved_path)
        return AnnOperator(OperatorType[obj.get('operator_type')], model)


    def support_features( self, *args ):
        return self.model.is_trained_on(*args)

    def __eq__(self, other):
        return super(AnnOperator, self).__eq__(other) and self.model.identifier == other.model.identifier

    def __str__(self):
        return self.model.identifier+' ['+str(self.operator_type)+']'