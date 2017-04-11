from .realNN import RealNN
import numpy as np
from .error import Unsolveable
from time import time

class Model:
    def __init__(self):
        self.constraints = []
        self.variables = []

        self.p = {}
        l = ['start_conf','leaving_entering']
        for pl in l:
            self.p[pl] = False

    def add_var(self, ty, name="NA", value=None):
        if ty == "real+":
            x = RealNN(name,value,index=len(self.variables))
            self.variables.append(x)
            return x

    def maximize(self, obj):
        self.obj_coefficients = obj.get_coefficients(len(self.variables))

    def add_constraint(self,constraint):
        self.constraints.append(constraint)

    def find_bfs(self):
        bfs_idx = []
        bfs_vars = []

        self.matrix = self.tableau[1:]
        self.bs = self.tableau[1:,-1]
        self.obj = self.tableau[0,:]

        self.row_to_var = [False for x in range(self.matrix.shape[0])]

        # Build slack variables

        identity = np.eye(self.matrix.shape[0])
        identity = np.r_[np.zeros((1,self.matrix.shape[0])), identity]
        self.tableau = np.c_[self.tableau[:,:-1], identity, self.tableau[:,-1]]

        self.matrix = self.tableau[1:]
        self.bs = self.tableau[1:,-1]
        self.obj = self.tableau[0,:]

        # range not including the b column
        # get all columns which have only one value => basis
        for c in range(len(self.variables),self.matrix.shape[1]-1):
            row = np.argwhere(self.matrix[:,c]!=0)[0][0]
            bfs_idx.append([row,c])
            bfs_vars.append(c)
            self.row_to_var[row] = c

        if len(bfs_idx) == self.matrix.shape[0]:
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

            # set obj

            self.tableau[0,:] = np.append(-np.array(self.obj_coefficients), np.zeros((1,self.matrix.shape[1]-len(self.variables))))

            if self.p['start_conf']:
                print("Tableau:")
                print(self.tableau)


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
                if len(positive):
                    entering = c
                    l = np.argmin(self.bs[positive]/self.matrix[positive,c])
                    leaving_row = positive[l]
                    leaving = self.row_to_var[leaving_row]

                    breaked = True
                    break
        if not breaked:
            return True

        if self.p['leaving_entering']:
            print("Leaving is %d" % leaving)
            print(self.matrix[leaving_row])
            print("Entering is %d" % entering)


        fac = -self.tableau[0,entering]/self.matrix[leaving_row,entering]
        self.tableau[0] = fac*self.matrix[leaving_row]+self.tableau[0]


        for row in range(self.matrix.shape[0]):
            if row != leaving_row:
                fac = -self.matrix[row,entering]/self.matrix[leaving_row,entering]
                self.matrix[row] = fac*self.matrix[leaving_row]+self.matrix[row]

        self.matrix[leaving_row] /= self.matrix[leaving_row,entering]

        self.new_basis(entering, leaving)
        i = 0
        for row in range(self.matrix.shape[0]):
            c = self.row_to_var[row]
            self.matrix[row] /= self.matrix[row,c]

            i += 1
        return False

    def print_solution(self):
        for c in range(len(self.variables)):
            if np.count_nonzero(self.matrix[:,c]) == 1:
                v_idx = np.where(self.matrix[:,c] == 1)[0]
                print("%s is %f" % (self.variables[c].name,self.bs[v_idx]))
        cc = 1
        for c in range(len(self.variables),self.matrix.shape[1]-1):
            if np.count_nonzero(self.matrix[:,c]) == 1:
                v_idx = np.where(self.matrix[:,c] == 1)[0]
                print("slack %s is %f" % (cc,self.bs[v_idx]))
            cc += 1
        print("Obj: %f" % (self.obj[-1]))

    def get_solution_object(self):
        sol_row = []

        for c in range(self.matrix.shape[1]-1):
            if np.count_nonzero(self.matrix[:,c]) == 1:
                v_idx = np.where(self.matrix[:,c] == 1)[0][0]
                sol_row.append(self.bs[v_idx])
            else:
                sol_row.append(0)
        sol_row.append(self.obj[-1])
        return sol_row

    def solve(self):

        self.tableau = np.zeros((len(self.constraints)+1,len(self.variables)+1))

        print("Solve")
        print("Information: ")
        print("We have %d variables " % len(self.variables))
        print("We have %d constraints " % len(self.constraints))
        i = 1
        for constraint in self.constraints:
            coefficients = constraint.x.get_coefficients(len(self.variables))
            self.tableau[i] = coefficients+[constraint.y]
            i += 1

            if constraint.type != "<=":
                raise Unsolveable("Type %s isn't accepted" % constraint.type)

        self.find_bfs()
        solved = self.pivot()
        steps = 1
        while not solved:
            solved = self.pivot()
            steps += 1
        print("End tableau")
        print(self.tableau)
        print("Steps: ", steps)

