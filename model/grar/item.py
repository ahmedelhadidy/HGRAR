class Item:

    def __init__(self, category, order):
        self.category = category
        self.order = order

    def value(self, data):
        return  data[self.category]

    def __str__(self):
        return self.category

    def __eq__(self, other):
        return type(self) == type(other) \
               and (self.category == other.category if (self.category and other.category) else self.order == other.order)


