from model.grar.operator import Operator, OperatorType
import model.grar.exceptions as exception
from math import e
from model.grar.exceptions import NotSupportedOperator


class ArithmeticOperator(Operator):

    def __init__(self, operator_type: OperatorType, is_fuzzy=False):
        if operator_type not in [OperatorType.GTE, OperatorType.LTE, OperatorType.NE]:
            raise exception.NotSupportedOperator('Operator Type %s Not Supported in class %s'.format(operator_type,type(self)))
        super().__init__(operator_type)
        self.is_fuzzy = is_fuzzy

    def revers(self):
        if self.operator_type == OperatorType.GTE:
            return ArithmeticOperator(OperatorType.LTE, self.is_fuzzy)
        elif self.operator_type == OperatorType.LTE:
            return ArithmeticOperator(OperatorType.GTE, self.is_fuzzy)
        else:
            return ArithmeticOperator(self.operator_type, self.is_fuzzy)

    def apply(self, value1, value2):
        if self.operator_type == OperatorType.GTE:
            return fuzzy_GTE(value1, value2) if self.is_fuzzy else GTH(value1, value2)
        if self.operator_type == OperatorType.LTE:
            return fuzzy_LTE(value1, value2) if self.is_fuzzy else LTE(value1, value2)
        if self.operator_type == OperatorType.EQ:
            return fuzzy_equal(value1, value2) if self.is_fuzzy else equal(value1, value2)
        raise NotSupportedOperator('Not supported operator %s'.format(self.operator_type.name))


    def __eq__(self, other):
        return type(self) == type(other) and self.operator_type == other.operator_type \
               and self.is_fuzzy == other.is_fuzzy

    def __str__(self):
        return str(self.operator_type)

def fuzzy_equal(x, y):
    r = 0.2
    m = 2.
    c = x if abs(x) >= abs(y) else y
    z = x if c == y else y
    s = r * (abs(x) + abs(y)) * 0.5
    a = (z - c) / s
    complex = e ** (-0.5 * a ** m)
    # return sqrt(complex.real**2 + complex.imag**2)
    return complex

def fuzzy_LTE(x, y):
    c = -1.5
    return 1 / (1 + e ** (-c * (x - y)))

def fuzzy_GTE(x, y):
    c = 1.5
    return 1 / (1 + e ** (-c * (x - y)))

def equal(x, y):
    return 1 if x == y else 0

def LTE(x, y):
    return 1 if x < y else 0

def GTH(x, y):
    return 1 if x > y else 0
