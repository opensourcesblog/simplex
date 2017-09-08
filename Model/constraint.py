import numpy as np

class Constraint:
    def __init__(self,x,ty,y):
        self.x = x
        self.y = y
        self.type = ty

    def __str__(self):
        return "Constraint: %s %s %s" % (self.x, self.type, self.y)


    def check_feasibility(self):
        value = self.x.value
        if value is None:
            return True
        else:
            if self.type == "<=":
                return self.value <= self.y

        return False

    def is_violated(self, dtype, values):
        x = np.array(self.x.get_coefficients(dtype,len(values)))
        val = np.sum(x*values)
        if self.type == "<=":
            if val > self.y:
                return True
        if self.type == ">=":
            if val < self.y:
                return True
        return False
