
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

