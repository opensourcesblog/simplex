from .realNN import RealNN
import numpy as np
from .error import *
from time import time

np.set_printoptions(precision=2,
                       threshold=100000,
                       linewidth=500, 
                       suppress=True)

class Model:
    def __init__(self, print_obj=False):
        self.constraints = []
        self.variables = []

        self.MINIMIZE = -1
        self.MAXIMIZE = 1

        if not print_obj:
            self.p = {}
        else:
            self.p = print_obj

        l = ['start_conf','leaving_entering']
        for pl in l:
            if pl not in self.p:
                self.p[pl] = False

    def add_var(self, ty, name="NA", value=None):
        if ty == "real+":
            x = RealNN(name,value,index=len(self.variables))
            self.variables.append(x)
            return x

    def maximize(self, obj):
        self.obj_coefficients = obj.get_coefficients(len(self.variables))
        self.obj_type = self.MAXIMIZE

    def minimize(self, obj):
        self.obj_coefficients = obj.get_coefficients(len(self.variables))
        self.obj_type = self.MINIMIZE

    def add_constraint(self,constraint):
        self.constraints.append(constraint)

    def redef_matrix_bs_obj(self):
        self.matrix = self.tableau[:-1]
        self.bs = self.tableau[:-1,-1]
        self.obj = self.tableau[-1,:]

    def find_bfs(self):

        self.redef_matrix_bs_obj()

        if self.obj_type == self.MINIMIZE:
            self.tableau = np.transpose(self.tableau)
            self.tableau[-1,:] = -self.tableau[-1,:]
            self.redef_matrix_bs_obj()

        self.row_to_var = [False for x in range(self.matrix.shape[0])]
        # Build slack variables
        identity = np.eye(self.matrix.shape[0])
        identity = np.r_[identity, np.zeros((1,self.matrix.shape[0]))]
        self.tableau = np.c_[self.tableau[:,:-1], identity, self.tableau[:,-1]]


        self.redef_matrix_bs_obj()

        # range not including the b column
        # get all columns which have only one value => basis
        row = 0
        for c in range(self.matrix.shape[1]-1-len(self.row_to_var),self.matrix.shape[1]-1):
            self.row_to_var[row] = c
            row += 1

        if self.p['start_conf']:
            print("Start Tableau:")
            print(self.tableau)


    def new_basis(self,entering,leaving):
        for row in range(self.matrix.shape[0]):
            if self.row_to_var[row] == leaving:
                self.row_to_var[row] = entering
                break

    def pivot(self):
        breaked = False

        # check if the current tableau is optimal
        # if optimal every value in obj is non negative
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
                else:
                    raise Unbounded(self.variables[c].name)
                    
        if not breaked:
            return True

        if self.p['leaving_entering']:
            print("Leaving is %d" % leaving)
            print(self.matrix[leaving_row])
            print("Entering is %d" % entering)

        # get the new objective function
        fac = -self.tableau[-1,entering]/self.matrix[leaving_row,entering]
        self.tableau[-1] = fac*self.matrix[leaving_row]+self.tableau[-1]



        # gausian elemination
        for row in range(self.matrix.shape[0]):
            if row != leaving_row:
                fac = -self.matrix[row,entering]/self.matrix[leaving_row,entering]
                self.matrix[row] = fac*self.matrix[leaving_row]+self.matrix[row]

        self.matrix[leaving_row] /= self.matrix[leaving_row,entering]


        # change basis variables
        self.new_basis(entering, leaving)

        return False

    def print_solution(self):
        if self.obj_type == self.MAXIMIZE:
            self.row_to_var = np.array(self.row_to_var)
            cor_to_variable = self.row_to_var < len(self.variables)
            for c in range(len(self.row_to_var)):
                if cor_to_variable[c]:
                    v_idx = self.row_to_var[c]
                    print("%s is %f" % (self.variables[v_idx].name,self.bs[c]))
                
            for c in range(len(self.row_to_var)):
                if not cor_to_variable[c]:
                    v_idx = self.row_to_var[c]
                    print("slack %d is %f" % ((v_idx-len(self.variables)+1),self.bs[c]))
            print("Obj: %f" % (self.obj[-1]))
        elif self.obj_type == self.MINIMIZE:
            v_idx = self.matrix.shape[1]-len(self.variables)-1
            for c in range(len(self.variables)):
                if self.obj[v_idx] != 0:
                    print("%s is %f" % (self.variables[c].name,self.obj[v_idx]))
                v_idx+=1
            cc = 1
            for c in range(self.matrix.shape[1]-len(self.variables)-1):
                if self.obj[c] != 0:
                    print("slack %s is %f" % (cc,self.obj[c]))
                cc += 1
            print("Obj: %f" % (self.obj[-1]))

    def get_solution_object(self):
        sol_row = []

        if self.obj_type == self.MAXIMIZE:
            for c in range(self.matrix.shape[1]-1):
                if np.count_nonzero(self.matrix[:,c]) == 1:
                    v_idx = np.where(self.matrix[:,c] == 1)[0][0]
                    sol_row.append(self.bs[v_idx])
                else:
                    sol_row.append(0)
            sol_row.append(self.obj[-1])
        elif self.obj_type == self.MINIMIZE:
            sol_row = np.append(self.obj[self.matrix.shape[0]+1:-1],self.obj[:self.matrix.shape[0]+1])
            sol_row = np.append(sol_row, self.obj[-1])
        return sol_row

    def solve(self):

        self.tableau = np.zeros((len(self.constraints)+1,len(self.variables)+1))

        print("Solve")
        print("Information: ")
        print("We have %d variables " % len(self.variables))
        print("We have %d constraints " % len(self.constraints))
        i = 0
        for constraint in self.constraints:
            coefficients = constraint.x.get_coefficients(len(self.variables))
            self.tableau[i] = coefficients+[constraint.y]
            i += 1

            if self.obj_type == self.MAXIMIZE:
                if constraint.type != "<=":
                    raise Unsolveable("Type %s isn't accepted" % constraint.type)
            if self.obj_type == self.MINIMIZE:
                if constraint.type != ">=":
                    raise Unsolveable("Type %s isn't accepted" % constraint.type)
        

        # set obj
        if self.obj_type == self.MINIMIZE:
            self.tableau[-1,:] = np.append(np.array(self.obj_coefficients), np.zeros((1,1)))
        elif self.obj_type == self.MAXIMIZE:
            self.tableau[-1,:] = np.append(-np.array(self.obj_coefficients), np.zeros((1,1)))


        self.find_bfs()
        solved = self.pivot()
        steps = 1
        while not solved:
            solved = self.pivot()
            steps += 1
        print("End tableau")
        print(np.around(self.tableau, decimals=1))
        print("Steps: ", steps)

