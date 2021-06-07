from model.grar.operator import OperatorType
from model.grar.operator import Operator
from NR.nural_network import Basic_NN
import model.grar.exceptions as exception
import numpy as np


class AnnOperator(Operator):

    def __init__(self, operator_type : OperatorType, model: Basic_NN):
        if operator_type not in [OperatorType.FAULTY, OperatorType.NON_FAULTY, OperatorType.NE]:
            raise exception.NotSupportedOperator('Operator Type %s Not Supported in class %s'.format(operator_type,type(self)))
        super().__init__(operator_type)
        self.model = model

    def apply(self, *values):
        prediction = self.model.predict_with_membership_degree( np.expand_dims(np.asarray(values),0))[0]
        member_ship = None
        if self.operator_type == OperatorType.FAULTY:
            member_ship = prediction[True]
        else:
            member_ship = prediction[False]
        return member_ship

    def revers(self):
        if self.operator_type == OperatorType.FAULTY:
            return AnnOperator(OperatorType.NON_FAULTY, self.model)
        else:
            return AnnOperator(OperatorType.FAULTY, self.model)
    def __str__(self):
        return self.model.identifier+' ['+str(self.operator_type)+']'