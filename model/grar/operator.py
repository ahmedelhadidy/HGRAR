
from enum import Enum, auto


class AutoOp(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class OperatorType(AutoOp):
    GTE = auto()
    LTE = auto()
    EQ = auto()
    FAULTY = auto()
    NON_FAULTY=auto()
    NE = auto()


class Operator:

    def __init__(self, operator_type: OperatorType):
        self.operator_type = operator_type

    def apply(self, *values):
        pass

    def revers(self):
        pass

    def create_object( self ):
        obj = {
            "class": "Operator",
            "operator_type": self.operator_type.value,
        }
        return obj

    def build_from_obj( obj ):
        return Operator(OperatorType[obj.get('operator_type')])

    def __str__(self):
        return str(self.operator_type)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self

    def __eq__(self, other):
        return self.operator_type == other.operator_type

