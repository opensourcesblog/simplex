from .realNN import RealNN
import numpy as np
from .error import *
from time import time
from .tableau import *

np.set_printoptions(precision=2,
                       threshold=1000000,
                       linewidth=600,
                       suppress=True)


class Model:
    def __init__(self, print_obj={}):
        self.constraints = []
        self.variables = []

        self.MINIMIZE = -1
        self.MAXIMIZE = 1
        self.DUAL = False

        self.DEF_DUAL = 2
        self.TEST_DUAL = 1
        self.NO_DUAL = 0

        self.t = TableauView()
        self.t.tableau = np.zeros((5,5))

        self.steps = 1
        self.t.nof_var_cols = 0

        default_print_obj = {
            'information': True,
            'start_conf': False,
            'leaving_entering': False,
            'end_conf': False,
            'timing': False
        }

        # This would work Python >= 3.5
        # self.p = {**default_print_obj, **print_obj}
        self.p = default_print_obj.copy()
        self.p.update(print_obj)

    def add_var(self, ty, name="NA", ub=None, lb=None, value=None):
        if ty == "real+":
            x = RealNN(name,value,index=len(self.variables))
            self.variables.append({"x": x,"ub": ub,"lb": lb})
            return x

    def maximize(self, obj):
        self.obj_coefficients = obj.get_coefficients(len(self.variables))
        self.obj_type = self.MAXIMIZE

    def minimize(self, obj):
        self.obj_coefficients = obj.get_coefficients(len(self.variables))
        self.obj_type = self.MINIMIZE

    def add_constraint(self,constraint):
        self.constraints.append(constraint)

    def add_new_col(self, col, obj):
        zeros = [0] * (self.t.tableau.shape[0] - 1 - len(col))
        new_variable_line = np.concatenate((col, zeros, [obj]))
        self.t.tableau = np.c_[
            self.t.tableau[:, :self.t.nof_var_cols], new_variable_line, self.t.tableau[:, self.t.nof_var_cols:]]

        self.row_to_var[self.row_to_var >= self.t.nof_var_cols] += 1
        self.t.nof_var_cols += 1

        # Update: c* = c-y*A
        A = self.t.A_star[:,-1, np.newaxis]
        self.t.c_star[-1] -= self.t.y_star.dot(A)
        self.t.c_star[-1] *= -1

        # Update: A*=S*A
        S_startA = self.t.S_star.dot(A).T[0]
        self.t.A_star[:,-1] = S_startA

        solved, _ = self.pivot()
        while not solved:
            solved, _ = self.pivot()
            self.steps += 1

    def add_new_variable(self, col, obj, name="NA", value=None):
        x = RealNN(name, value, index=len(self.variables))
        self.variables.append({"x": x})
        if self.DUAL:
            zeros = [0] * (self.t.tableau.shape[1] - 1 - self.t.nof_var_cols)
            new_row = np.array(col + zeros + [obj])
            if self.obj_type == self.MINIMIZE:
                new_row *= -1
            new_row = new_row[np.newaxis, :]
            self.add_new_row(new_row)
        else:
            if self.obj_type == self.MINIMIZE:
                self.add_new_col(-np.array(col), -obj)
            else:
                self.add_new_col(col,obj)

    def add_new_row(self, new_row):
        self.t.tableau = np.r_[self.t.tableau[:-1], new_row, self.t.tableau[-1, np.newaxis]]

        # adjusting the matrix
        # add slack variable
        one_slack = np.zeros((self.t.tableau.shape[0],1))
        one_slack[-2] = 1
        self.t.tableau = np.c_[self.t.tableau[:,:-1], one_slack, self.t.tableau[:,-1][:,np.newaxis]]

        # A_m+1=A_m+1-A_(m+1,B)*A*
        A = self.t.matrix
        A_m1 = A[-1]
        A_m1B = A_m1[self.row_to_var]
        self.t.A_star[-1] -= A_m1B.dot(self.t.A_star[:-1])

        # S_m+1=-A_(m+1,B)*S*
        self.t.S_star[-1,:-1] = -A_m1B.dot(self.t.S_star[:-1,:-1])

        # b_m+1 = b_m+1-A_(m+1,B)b*
        self.t.bs[-1] -= A_m1B.dot(self.t.bs[:-1])

        # check if b is negative => not feasible => dual pivot
        self.row_to_var = np.append(self.row_to_var, self.t.tableau.shape[1]-2)
        if self.t.bs[-1] >= 0:
            return

        # Dual pivot
        while np.any(self.t.bs < 0):
            row = np.argmin(self.t.bs)
            a_m1 = np.copy(self.t.tableau[row, :-1])
            a_m1[a_m1 >= 0] = 0
            if np.any(a_m1 < 0):
                quot = np.argmin(np.abs(divInf(self.t.obj[:-1], a_m1)))
                self.gauss_step(row, quot)
            else:
                raise InfeasibleError("The model is infeasible")

        solved, _ = self.pivot()
        while not solved:
            solved, _ = self.pivot()
            self.steps += 1

    def add_lazy_constraint(self, constraint):
        self.constraints.append(constraint)
        if self.DUAL:
            # add constraint as new variable
            coefficients = np.array(constraint.x.get_coefficients(len(self.variables)))
            if constraint.type != ">=":
                coefficients *= -1
                constraint.y *= -1
            self.add_new_col(coefficients, constraint.y)
        else:
            coefficients = constraint.x.get_coefficients(len(self.variables))
            zeros = [0] * (self.t.tableau.shape[1] - 1 - len(self.variables))
            new_constraint_line = np.array(coefficients + zeros + [constraint.y])
            if constraint.type != "<=":
                new_constraint_line *= -1
            new_constraint_line = new_constraint_line[np.newaxis, :]
            self.add_new_row(new_constraint_line)

    def to_dual(self):
        self.t.tableau = self.t.tableau.T
        # number of cols for variables changed =>
        self.t.nof_var_cols = self.t.tableau.shape[1]-1
        self.obj_type = self.MAXIMIZE if self.obj_type == self.MINIMIZE else self.MINIMIZE

    def to_standard(self):
        if self.obj_type == self.MINIMIZE:
            self.t.tableau *= -1
        if np.any(self.t.bs < 0):
            # solve start problem
            first_obj = -np.copy(self.t.obj)
            # add x_0 variable
            ones = np.ones(self.t.matrix.shape[0]+1)
            self.t.tableau = np.c_[self.t.tableau[:, :-1], -ones, self.t.tableau[:, -1]]

            # and change obj to max -x_0
            self.t.tableau[-1,:] = np.zeros(self.t.tableau.shape[1])
            self.t.tableau[-1,-2] = 1

            self.find_bfs()
            # get biggest b
            row = np.argmin(self.t.bs)
            self.gauss_step(row,self.t.nof_var_cols)
            solved, _ = self.pivot()
            while not solved:
                solved, _ = self.pivot()
                self.steps += 1

            # if x_0 is 0 => we have a bfs for our real maximization problem
            if self.t.obj[-1] == 0:
                # remove the x_0 column
                self.t.tableau = np.c_[self.t.tableau[:,:self.t.nof_var_cols],self.t.tableau[:,self.t.nof_var_cols+1:]]

                # update self.row_to_var
                self.row_to_var[self.row_to_var > self.t.nof_var_cols] -= 1

                rows = np.where(self.row_to_var < self.t.nof_var_cols)[0]
                vs = []
                for row in rows:
                    v = self.row_to_var[row]
                    self.t.obj += first_obj[v]*self.t.tableau[row]
                    vs.append(v)
                for v in range(self.t.nof_var_cols):
                    if v not in vs:
                        self.t.obj[v] -= first_obj[v]
                        pass
                for v in vs:
                    self.t.obj[v] = 0

            else:
                raise Unsolveable("Wasn't possible to form the standard form")
        else:
            self.find_bfs()

    def find_bfs(self):
        self.row_to_var = [False for x in range(self.t.matrix.shape[0])]
        # Build slack variables
        identity = np.eye(self.t.matrix.shape[0])
        identity = np.r_[identity, np.zeros((1,self.t.matrix.shape[0]))]
        self.t.tableau = np.c_[self.t.tableau[:,:-1], identity, self.t.tableau[:,-1]]

        # range not including the b column
        # get all columns which have only one value => basis
        row = 0
        for c in range(self.t.matrix.shape[1]-1-len(self.row_to_var),self.t.matrix.shape[1]-1):
            self.row_to_var[row] = c
            row += 1
        self.row_to_var = np.array(self.row_to_var)

        if self.p['start_conf']:
            print("Start Tableau:")
            print(self.t.tableau)


    def new_basis(self,entering,leaving):
        for row in range(self.t.matrix.shape[0]):
            if self.row_to_var[row] == leaving:
                self.row_to_var[row] = entering
                break

    def pivot(self):
        # check if the current tableau is optimal
        # if optimal every value in obj is non negative
        get_l_e = time()
        min_not_in_basis = np.copy(self.t.obj[:-1])
        min_not_in_basis[self.row_to_var] = 0 # non negative
        c = np.argmin(min_not_in_basis)
        get_l_e = time() - get_l_e
        if self.t.obj[c] < 0:
            positive = np.where(self.t.matrix[:,c] > 0)[0]
            if len(positive):
                entering = c
                l = np.argmin(self.t.bs[positive]/self.t.matrix[positive,c])
                leaving_row = positive[l]
                leaving = self.row_to_var[leaving_row]
            else:
                raise Unbounded(self.variables[c]["x"].name)
        else:
            # get_l_e = time()-get_l_e
            return True, get_l_e

        # get_l_e = time()-get_l_e

        if self.p['leaving_entering']:
            print("Leaving is %d" % leaving)
            print(self.t.matrix[leaving_row])
            print("Entering is %d" % entering)

        self.gauss_step(leaving_row, entering)

        return False, get_l_e

    def gauss_step(self, leaving_row, entering):
        # get the new objective function
        fac = -self.t.tableau[-1, entering] / self.t.matrix[leaving_row, entering]
        self.t.tableau[-1] = fac * self.t.matrix[leaving_row] + self.t.tableau[-1]

        # gausian elimination
        facs = -self.t.matrix[:, entering] / self.t.matrix[leaving_row, entering]
        facs[leaving_row] = 0

        self.t.matrix += facs[:, np.newaxis] * self.t.matrix[leaving_row, np.newaxis, :]

        self.t.matrix[leaving_row] /= self.t.matrix[leaving_row, entering]

        self.t.tableau[:,entering] = 0
        self.t.tableau[leaving_row,entering] = 1

        # change basis variables
        leaving = self.row_to_var[leaving_row]
        self.new_basis(entering, leaving)

    def check_if_dual(self):
        if self.DUAL:
            return
        if self.obj_type == self.MINIMIZE:
            if len(self.variables)/8 < len(self.constraints):
                self.DUAL = True

    def print_solution(self,slack=False):
        if self.DUAL:
            for c in range(self.t.nof_var_cols,self.t.nof_var_cols+len(self.variables)):
                v_idx = c-self.t.nof_var_cols
                if self.t.obj[c] > 0:
                    print("%s is %f" % (self.variables[v_idx]["x"].name, self.t.obj[c]))
        else:
            cor_to_variable = self.row_to_var < len(self.variables)
            for c in range(len(self.row_to_var)):
                if cor_to_variable[c]:
                    v_idx = self.row_to_var[c]
                    print("%s is %f" % (self.variables[v_idx]["x"].name,self.t.bs[c]))
        if slack:
            if self.DUAL:
                print("Problem solved using dual simplex => no slack variables possible")
            else:
                for c in range(len(self.row_to_var)):
                    if not cor_to_variable[c]:
                        v_idx = self.row_to_var[c]
                        print("slack %d is %f" % ((v_idx-len(self.variables)+1),self.t.bs[c]))
        if self.obj_type == self.MAXIMIZE:
            print("Obj: %f" % (self.t.obj[-1]))
        elif self.obj_type == self.MINIMIZE:
            print("Obj: %f" % (-self.t.obj[-1]))

    def get_solution_object(self):
        if self.DUAL:
            sol_row = list(self.t.obj[self.t.nof_var_cols:-1])
        else:
            cor_to_variable = self.row_to_var < len(self.variables)
            sol_row = [0]*len(self.variables)
            for c in range(len(self.row_to_var)):
                if cor_to_variable[c]:
                    v_idx = self.row_to_var[c]
                    sol_row[v_idx] = self.t.bs[c]
        if self.obj_type == self.MAXIMIZE:
            sol_row.append(self.t.obj[-1])
        elif self.obj_type == self.MINIMIZE:
            sol_row.append(-self.t.obj[-1])

        return sol_row


    def solve(self, consider_dual=None):
        if consider_dual is None:
            consider_dual = self.TEST_DUAL

        self.t.tableau = np.zeros((len(self.constraints)+1,len(self.variables)+1))

        if self.p['information']:
            print("Information: ")
            print("We have %d variables " % len(self.variables))
            print("We have %d constraints " % len(self.constraints))

        self.t.nof_var_cols = len(self.variables)
        i = 0
        for constraint in self.constraints:
            coefficients = constraint.x.get_coefficients(len(self.variables))
            # if constraint.y < 0:
            #     raise Unsolveable("All constants on the right side must be non-negative")
            self.t.tableau[i] = coefficients+[constraint.y]

            if self.obj_type == self.MAXIMIZE:
                if constraint.type != "<=":
                    self.t.tableau[i] *= -1
            if self.obj_type == self.MINIMIZE:
                if constraint.type != ">=":
                    self.t.tableau[i] *= -1
            i += 1

        # set obj
        self.t.tableau[-1,:] = np.append(np.array(self.obj_coefficients), np.zeros((1,1)))
        if consider_dual == self.DEF_DUAL:
            self.DUAL = True
        if consider_dual == self.TEST_DUAL:
            self.check_if_dual()
        if self.DUAL:
            if self.p['information']:
                print("Using dual")
            self.to_dual()

        self.t.tableau[-1, :] = -self.t.tableau[-1,:]

        t_to_standard = time()
        self.to_standard()
        t_to_standard = time()-t_to_standard

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
            print("To standard time", t_to_standard)
            print("Steps in last part", this_steps)
            print("Total time: ", total)
            print("per step", total/this_steps)
            print("Total le time: ", total_l_e)
            print("per step le", total_l_e / this_steps)

        if self.p['end_conf']:
            print("End tableau")
            print(np.around(self.t.tableau, decimals=4))
            print("Steps: ", self.steps)


def divInf(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [Inf, Inf, Inf] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[np.isnan(c)] = np.inf
    return c
