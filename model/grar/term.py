from model.grar.item import Item
from model.grar.operator import Operator


class Term:

    def __init__(self, item: Item, operator: Operator):
        self.item = item
        self.operator = operator

    def apply(self, right_hand_side_item: Item, data_row):
        value1 = self.item.value(data_row)
        value2 = right_hand_side_item.item.value(data_row)
        return self.operator.apply(value1, value2)

    def get_reversed(self):
        return Term(self.item, self.operator.revers())

    def __str__(self):
        return str(self.item) +" "+ str(self.operator)+" "

    def __eq__(self, other):
        return type(self) == type(other) and  self.item == other.item and self.operator == other.operator
