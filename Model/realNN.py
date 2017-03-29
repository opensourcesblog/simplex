from .constraint import Constraint
from .error import NonLinear
from .list_RealNN import List_RealNN

class RealNN:
    def __init__(self,name=None,value=None,index=0,factor=1):
        self.name = name
        self.value = value
        self.factor = factor
        self.index = index

    def __str__(self):
        if self.value is None:
            return "%s has no value" % self.name
        else:
            return "%s has value %.2f" % (self.name,self.value)

    def __eq__(self, other):
        return Constraint(self, "==", other)

    def __le__(self, other):
        return Constraint(self, "<=", other)

    def __add__(self, other):
        if self.value is not None and other.value is not None:
            return RealNN(value=self.factor*self.value+other.factor*other.value)
        else:
            return List_RealNN(self,other)

    def __rmul__(self, factor):
        if isinstance(factor, (int, float, complex)):
            return RealNN(self.name,self.value,index=self.index,factor=factor)
        else:
            raise NonLinear("factor %s is not linear" % factor)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
