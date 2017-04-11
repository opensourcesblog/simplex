from .constraint import Constraint
from .error import NonLinear

class List_RealNN:
    def __init__(self,realNN_x,realNN_y):
        from .realNN import RealNN

        if isinstance(realNN_x, (RealNN)):
            self.list = [realNN_x,realNN_y]
        elif isinstance(realNN_x, (List_RealNN)):
            self.list = realNN_x.list
            self.list.append(realNN_y)

    def get_coefficients(self,l=False):
        if not l:
            l = len(self.list)
        l_factor = [0]*l
        for realNN in self.list:
            l_factor[realNN.index] = realNN.factor
        return l_factor

    def __eq__(self, other):
        return Constraint(self, "==", other)

    def __le__(self, other):
        return Constraint(self, "<=", other)

    def __add__(self, other):
        return List_RealNN(self,other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
