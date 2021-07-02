class Item:

    def __init__(self, category, order):
        self.category = category
        self.order = order

    def value(self, data):
        return  data[self.category]

    def get_identifier( self ):
        return self.category if self.category else self.order

    def create_object( self ):
        obj = {
            'category': self.category,
            'order': self.order
        }
        return obj

    def build_from_obj( obj ):
        category = obj.get('category', None)
        order = obj.get('order', None)
        return Item(category, order)


    def __str__(self):
        return self.category

    def __eq__(self, other):
        return type(self) == type(other) and self.get_identifier() == other.get_identifier()


