from model.grar.operator import Operator, OperatorType
from model.grar.operator import Operator
from NR.nr import ParentNR
import model.grar.exceptions as exception


class AnnOperator(Operator):

    def __init__(self, operator_type : OperatorType, model: ParentNR):
        if operator_type not in [OperatorType.FAULTY, OperatorType.NON_FAULTY, OperatorType.NE]:
            raise exception.NotSupportedOperator('Operator Type %s Not Supported in class %s'.format(operator_type,type(self)))
        super().__init__(operator_type)
        self.model = model

    def apply(self, *values):
        prediction = self.model.predict_with_membership_degree([values])[0]
        if self.operator_type == OperatorType.FAULTY:
            return prediction[True]
        else:
            return prediction[False]

    def revers(self):
        if self.operator_type == OperatorType.FAULTY:
            return AnnOperator(OperatorType.NON_FAULTY, self.model)
        else:
            return AnnOperator(OperatorType.FAULTY, self.model)
    def __str__(self):
        return self.model.identifier+' ['+str(self.operator_type)+']'