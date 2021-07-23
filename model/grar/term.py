from model.grar.item import Item
from model.grar.operator import Operator
from model.grar.perceptron_operator import AnnOperator
from model.grar.operator import OperatorType


class Term:

    def __init__(self, item: Item, operator: Operator):
        self.item = item
        self.operator = operator

    def apply(self, right_hand_side_item: Item, data_row):
        values_obj={}
        value1 = self.item.value(data_row)
        value2 = right_hand_side_item.item.value(data_row)
        values_obj[self.item.get_identifier()] = value1
        values_obj[right_hand_side_item.item.get_identifier()] = value2
        return self.operator.apply(values_obj)[0]

    def apply_bulk(self, right_hand_side_item: Item, dataset):
        objects=[]
        for index, data_row in dataset.iterrows():
            values_obj={}
            value1 = self.item.value(data_row)
            value2 = right_hand_side_item.item.value(data_row)
            values_obj[self.item.get_identifier()] = value1
            values_obj[right_hand_side_item.item.get_identifier()] = value2
            objects.append(values_obj)
        return self.operator.apply(*objects)

    def get_reversed(self):
        return Term(self.item, self.operator.revers())

    def create_object( self ):
        obj = {
            "item": self.item.create_object(),
            "operator": self.operator.create_object()
        }
        return obj

    def build_from_obj(  obj ):
        item = Item.build_from_obj (obj.get('item'))
        operator_class = obj.get('operator').get('class')
        if operator_class == 'AnnOperator':
            operator = AnnOperator.create_from_obj(obj.get('operator'))
        else:
            operator = Operator.build_from_obj(obj.get('operator'))
        return Term(item, operator)

    def __str__(self):
        return str(self.item) +" "+ str(self.operator)+" "

    def __eq__(self, other):
        return type(self) == type(other) and  self.item == other.item and self.operator == other.operator

    def grar_eq( self, other ):
        return self.item == other.item and (self.operator == other.operator or OperatorType.NE in (self.operator.operator_type, other.operator.operator_type))

