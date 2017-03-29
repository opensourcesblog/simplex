from .realNN import RealNN
import numpy as np

class Model:
    def __init__(self):
        self.constraints = []
        self.variables = []

    def add_var(self, ty, name="NA", value=None):
        if ty == "real+":
            x = RealNN(name,value,index=len(self.variables))
            self.variables.append(x)
            return x

    def minimize(self, obj):
        pass

    def add_constraint(self,constraint):
        self.constraints.append(constraint)

    def solve(self):
        self.tableau = np.zeros((len(self.constraints)+1,len(self.variables)+1))

        print("Solve")
        print("Information: ")
        print("We have %d variables " % len(self.variables))
        print("We have %d constraints " % len(self.constraints))
        print("List of constraints:")
        i = 1
        for constraint in self.constraints:
            coefficients = constraint.x.get_coefficients(len(self.variables))
            self.tableau[i] = coefficients+[constraint.y]
            i += 1

            print("Left side: ", coefficients)
            print("Type: ", constraint.type)
            print("Right side: ", constraint.y)
        print("Tableau")
        print(self.tableau)
