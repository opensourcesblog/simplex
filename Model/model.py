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

    def find_bfs(self):
        bfs_idx = []
        bfs_vars = []
        self.row_to_var = [False for x in range(self.matrix.shape[0])]
        # range not including the b column
        # get all columns which have only one value => basis
        for c in range(self.matrix.shape[1]-1):
            if np.count_nonzero(self.matrix[:,c]) == 1:
                row = np.argwhere(self.matrix[:,c]!=0)[0][0]
                print("row",row)
                bfs_idx.append([row,c])
                bfs_vars.append(c)
                self.row_to_var[row] = c

        if len(bfs_idx) == self.matrix.shape[0]:
            print("Found bfs")
            print("idx", bfs_idx)
            # build up the objective function
            # first devide by value in matrix to get the identity matrix
            # then get the objective function in terms of the non basian variables
            i = 0
            for row in range(self.matrix.shape[0]):
                col = self.row_to_var[row]
                val_in_matrix = self.matrix[row,col]
                # get an identity matrix
                self.matrix[row] /= val_in_matrix
                i += 1
            self.new_obj()

            print("Tableau:")
            print(self.tableau)
            print("Bs:")
            print(self.bs)

    def new_obj(self):
        for c in range(self.matrix.shape[1]):
            if c not in self.row_to_var:
                self.obj[c] = -np.sum(self.matrix[:,c])
            else:
                self.obj[c] = 0


    def new_basis(self,entering,leaving):
        for row in range(self.matrix.shape[0]):
            if self.row_to_var[row] == leaving:
                self.row_to_var[row] = entering
                break

    def pivot(self):
        breaked = False
        for c in range(self.matrix.shape[1]-1):
            if c in self.row_to_var:
                continue
            if self.obj[c] < 0:
                positive = np.where(self.matrix[:,c] > 0)[0]
                print("positive: ",positive)
                if len(positive):
                    entering = c
                    print(-self.bs[positive]/self.matrix[positive,c])
                    l = np.argmin(self.bs[positive]/self.matrix[positive,c])
                    leaving_row = positive[l]
                    leaving = self.row_to_var[leaving_row]

                    breaked = True
                    break
        if not breaked:
            return True
        print("Leaving is %d" % leaving)
        print(self.matrix[leaving_row])
        print("Entering is %d" % entering)


        self.matrix[leaving_row] /= self.matrix[leaving_row,entering]
        for row in range(self.matrix.shape[0]):
            if row != leaving_row:
                self.matrix[row] /= self.matrix[row,entering]
                self.matrix[row] -= self.matrix[leaving_row]

        self.new_basis(entering, leaving)
        print("new basis")
        print(self.tableau)
        i = 0
        for row in range(self.matrix.shape[0]):
            c = self.row_to_var[row]
            self.matrix[row] /= self.matrix[row,c]

            i += 1
        self.new_obj()
        return False

    def print_solution(self):
        for c in range(self.matrix.shape[1]):
            if np.count_nonzero(self.matrix[:,c]) == 1:
                v_idx = np.where(self.matrix[:,c] == 1)[0]
                print("%s is %f" % (self.variables[c].name,self.bs[v_idx]))
        print("Obj: %f" % (-self.obj[-1]))

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
        self.matrix = self.tableau[1:]
        self.bs = self.tableau[1:,-1]
        self.obj = self.tableau[0,:]
        self.find_bfs()
        solved = self.pivot()
        while not solved:
            solved = self.pivot()
        print("End tableau")
        print(self.tableau)
