from .realNN import RealNN
import numpy as np
from .error import *
from time import time

np.set_printoptions(precision=2,
                       threshold=100000,
                       linewidth=600,
                       suppress=True)


class Model:
    def __init__(self, print_obj=False):
        self.constraints = []
        self.variables = []

        self.MINIMIZE = -1
        self.MAXIMIZE = 1
        self.steps = 1

        if print_obj is False:
            self.p = {'information': True}
        else:
            self.p = print_obj

        l = ['start_conf','information','leaving_entering','end_conf', 'timing']
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

    def to_standard(self):
        if self.obj_type == self.MINIMIZE:
            self.tableau *= -1
        self.redef_matrix_bs_obj()
        if np.any(self.bs < 0):
            # solve start problem
            first_obj = -np.copy(self.obj)
            # add x_0 variable
            ones = np.ones(self.matrix.shape[0]+1)
            self.tableau = np.c_[self.tableau[:, :-1], -ones, self.tableau[:, -1]]

            # and change obj to max -x_0
            self.tableau[-1,:] = np.zeros(self.tableau.shape[1])
            self.tableau[-1,-2] = 1
            self.redef_matrix_bs_obj()

            self.find_bfs()
            # get biggest b
            row = np.argmin(self.bs)
            self.gauss_step(row,len(self.variables))
            solved, _ = self.pivot()
            while not solved:
                solved, _ = self.pivot()
                self.steps += 1

            # if x_0 is 0 => we have a bfs for our real maximization problem
            if self.obj[-1] == 0:
                # remove the x_0 column
                self.tableau = np.c_[self.tableau[:,:len(self.variables)],self.tableau[:,len(self.variables)+1:]]
                self.redef_matrix_bs_obj()
                rows = np.where(self.row_to_var < len(self.variables))[0]
                vs = []
                for row in rows:
                    v = self.row_to_var[row]
                    self.obj += first_obj[v]*self.tableau[row]
                    vs.append(v)
                for v in range(len(self.variables)):
                    if v not in vs:
                        self.obj[v] -= first_obj[v]
                        pass
                for v in vs:
                    self.obj[v] = 0

            else:
                raise Unsolveable("Wasn't possible to form the standard form")
        else:
            self.find_bfs()

    def find_bfs(self):
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
        self.row_to_var = np.array(self.row_to_var)

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
        get_l_e = time()
        for c in np.argsort(self.obj[:-1]):
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
        get_l_e = time()-get_l_e
        if not breaked:
            return True, get_l_e


        if self.p['leaving_entering']:
            print("Leaving is %d" % leaving)
            print(self.matrix[leaving_row])
            print("Entering is %d" % entering)

        self.gauss_step(leaving_row, entering)

        return False, get_l_e

    def gauss_step(self, leaving_row, entering):
        # get the new objective function
        fac = -self.tableau[-1, entering] / self.matrix[leaving_row, entering]
        self.tableau[-1] = fac * self.matrix[leaving_row] + self.tableau[-1]

        # gausian elimination
        facs = -self.matrix[:, entering] / self.matrix[leaving_row, entering]
        facs[leaving_row] = 0

        self.matrix += facs[:, np.newaxis] * self.matrix[leaving_row, np.newaxis, :]

        self.matrix[leaving_row] /= self.matrix[leaving_row, entering]

        # change basis variables
        leaving = self.row_to_var[leaving_row]
        self.new_basis(entering, leaving)


    def print_solution(self,slack=False):
        self.row_to_var = np.array(self.row_to_var)
        cor_to_variable = self.row_to_var < len(self.variables)
        for c in range(len(self.row_to_var)):
            if cor_to_variable[c]:
                v_idx = self.row_to_var[c]
                print("%s is %f" % (self.variables[v_idx].name,self.bs[c]))
        if slack:
            for c in range(len(self.row_to_var)):
                if not cor_to_variable[c]:
                    v_idx = self.row_to_var[c]
                    print("slack %d is %f" % ((v_idx-len(self.variables)+1),self.bs[c]))
        if self.obj_type == self.MAXIMIZE:
            print("Obj: %f" % (self.obj[-1]))
        elif self.obj_type == self.MINIMIZE:
            print("Obj: %f" % (-self.obj[-1]))

    def get_solution_object(self):
        sol_row = []
        for c in range(self.matrix.shape[1]-1):
            if np.count_nonzero(self.matrix[:,c]) == 1:
                v_idx = np.where(self.matrix[:,c] == 1)[0][0]
                sol_row.append(self.bs[v_idx])
            else:
                sol_row.append(0)
        if self.obj_type == self.MAXIMIZE:
            sol_row.append(self.obj[-1])
        elif self.obj_type == self.MINIMIZE:
            sol_row.append(-self.obj[-1])

        return sol_row

    def solve(self):

        self.tableau = np.zeros((len(self.constraints)+1,len(self.variables)+1))

        if self.p['information']:
            print("Information: ")
            print("We have %d variables " % len(self.variables))
            print("We have %d constraints " % len(self.constraints))
        i = 0
        for constraint in self.constraints:
            coefficients = constraint.x.get_coefficients(len(self.variables))
            # if constraint.y < 0:
            #     raise Unsolveable("All constants on the right side must be non-negative")
            self.tableau[i] = coefficients+[constraint.y]


            if self.obj_type == self.MAXIMIZE:
                if constraint.type != "<=":
                    self.tableau[i] *= -1
            if self.obj_type == self.MINIMIZE:
                if constraint.type != ">=":
                    self.tableau[i] *= -1
            i += 1

        # set obj
        self.tableau[-1,:] = np.append(-np.array(self.obj_coefficients), np.zeros((1,1)))

        self.to_standard()

        solved, tim_l_e = self.pivot()
        total = 0
        this_steps = 0
        total_l_e = 0
        while not solved:
            s = time()
            solved, tim_l_e = self.pivot()
            total += time()-s
            total_l_e += tim_l_e
            this_steps += 1
            self.steps += 1

        if self.p['timing'] and this_steps > 0:
            print("Steps in last part", this_steps)
            print("Total time: ", total)
            print("per step", total/this_steps)
            print("Total le time: ", total_l_e)
            print("per step le", total_l_e / this_steps)

        if self.p['end_conf']:
            print("End tableau")
            print(np.around(self.tableau, decimals=1))
            print("Steps: ", self.steps)

