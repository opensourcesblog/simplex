from .constraint import Constraint
from .error import NonLinear
from .list_Var import List_Var


class Var:
    def __init__(self,name=None,value=None,type=None,index=0,factor=1):
        self.name = name
        self.value = value
        self.factor = factor
        self.index = index
        self.type = type

    def __str__(self):
        if self.value is None:
            return "%s" % self.name
        else:
            return "%s has value %.2f" % (self.name,self.value)

    def __eq__(self, other):
        return Constraint(self, "==", other)

    def __le__(self, other):
        return Constraint(self, "<=", other)

    def __ge__(self, other):
        return Constraint(self, ">=", other)

    def __add__(self, other):
        if self.value is not None and other.value is not None:
            return Var(value=self.factor * self.value + other.factor * other.value)
        else:
            return List_Var(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __neg__(self):
        return Var(name=self.name, value=self.value, index=self.index, factor=-self.factor)

    def __sub__(self, other):
        if self.value is not None and other.value is not None:
            return Var(value=self.factor * self.value - other.factor * other.value)
        else:
            return List_Var(self, -other)

    def __rsub__(self, other):
        if other == 0:
            return self
        else:
            return self.__sub__(other)

    def __rmul__(self, factor):
        if isinstance(factor, (int, float)):
            return Var(self.name, self.value, index=self.index, factor=factor)
        else:
            raise NonLinear("factor %s is not linear" % factor)



    def get_coefficients(self,l=False):
        if not l:
            l = 1
        l_factor = [0]*l
        l_factor[self.index] = self.factor
        return l_factor
