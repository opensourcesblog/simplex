from .constraint import Constraint
from .error import NonLinear
import  numpy as np
from fractions import Fraction


class List_Var:
    def __init__(self, var_x, var_y=False):
        from .Var import Var
        if var_y is False:
            self.list = var_x if isinstance(var_x, list) else [var_x]
        elif isinstance(var_x, (Var)):
            if isinstance(var_y, (Var)):
                self.list = [var_x, var_y]
            else:
                self.list = [var_x]+var_y.list
        elif isinstance(var_x, (List_Var)):
            if isinstance(var_y, (Var)):
                self.list = var_x.list+ [var_y]
            else:
                self.list = var_x.list+var_y.list

    def get_coefficients(self,dtype="float",l=False):
        if not l:
            l = len(self.list)
        l_factor = [0]*l
        if dtype == "fraction":
            for var in self.list:
                l_factor[var.index] = Fraction(var.factor)
        else:
            for var in self.list:
                l_factor[var.index] = var.factor
        return l_factor

    def __str__(self):
        return str(self.get_coefficients())

    def __neg__(self):
        return List_Var([-x for x in self.list])

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

