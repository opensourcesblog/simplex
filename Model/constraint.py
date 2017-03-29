
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

