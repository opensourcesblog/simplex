from .constraint import Constraint
from .error import NonLinear
import  numpy as np
from fractions import Fraction


class List_Var:
    def __init__(self, var_x, var_y):
        from .Var import Var

        if isinstance(var_x, (Var)):
            self.list = [var_x, var_y]
        elif isinstance(var_x, (List_Var)):
            self.list = var_x.list
            self.list.append(var_y)

    def get_coefficients(self,l=False):
        if not l:
            l = len(self.list)
        l_factor = [0]*l
        for var in self.list:
            l_factor[var.index] = Fraction(var.factor)
        return l_factor

    def __eq__(self, other):
        return Constraint(self, "==", other)

    def __le__(self, other):
        return Constraint(self, "<=", other)

    def __ge__(self, other):
        return Constraint(self, ">=", other)

    def __add__(self, other):
        return List_Var(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        return List_Var(self, -other)

    def __rsub__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(-other)

