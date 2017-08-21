from .Var import Var
import numpy as np
from .error import *
from time import time
from .tableau import *
from .bnb import BnB
from .bnbbp import BnBBP
from fractions import Fraction

np.set_printoptions(precision=2,
                       threshold=1000000,
                       linewidth=600,
                       suppress=True)


class Model:
    def __init__(self, print_obj={}):
        self.MINIMIZE = -1
        self.MAXIMIZE = 1

        self.DEF_DUAL = 2
        self.TEST_DUAL = 1
        self.NO_DUAL = 0

        self.t = TableauView()
        self.t.tableau = np.zeros((1,1), dtype=object)

        self.steps = 1
        self.t.nof_var_cols = 0
        self.t.Dual = False
        self.t.obj_type = self.MAXIMIZE
        self.t.type = self.MAXIMIZE
        self.t.constraints = []
        self.t.variables = []

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
        if ty == "real+" or ty == "int+":
            x = Var(name, value, type=ty, index=len(self.t.variables))
            self.t.variables.append({"x": x,"ub": ub,"lb": lb})
            return x
        else:
            raise TypeError("Only real and int values are accepted as variable types")

    def maximize(self, obj):
        self.t.obj_coefficients = obj.get_coefficients(len(self.t.variables))
        self.t.type = self.MAXIMIZE
        self.t.obj_type = self.MAXIMIZE

    def minimize(self, obj):
        self.t.obj_coefficients = obj.get_coefficients(len(self.t.variables))
        self.t.type = self.MINIMIZE
        self.t.obj_type = self.MINIMIZE

    def add_constraint(self,constraint):
        self.t.constraints.append(constraint)

    def add_new_col(self, col, obj):
        zeros = [0] * (self.t.tableau.shape[0] - 1 - len(col))
        new_variable_line = np.concatenate((col, zeros, [obj]))
        self.t.tableau = np.c_[
            self.t.tableau[:, :self.t.nof_var_cols], new_variable_line, self.t.tableau[:, self.t.nof_var_cols:]]

        self.t.row_to_var[self.t.row_to_var >= self.t.nof_var_cols] += 1
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
        x = Var(name, value, index=len(self.t.variables))
        self.t.variables.append({"x": x})
        if self.t.dual:
            zeros = [0] * (self.t.tableau.shape[1] - 1 - self.t.nof_var_cols)
            new_row = np.array(col + zeros + [obj])
            if self.t.type == self.MINIMIZE:
                new_row *= -1
            new_row = new_row[np.newaxis, :]
            self.add_new_row(new_row)
        else:
            if self.t.type == self.MINIMIZE:
                self.add_new_col(-np.array(col), -obj)
            else:
                self.add_new_col(col,obj)

    def add_new_row(self, new_row):
        self.t.tableau = np.r_[self.t.tableau[:-1], new_row, self.t.tableau[-1, np.newaxis]]

        # adjusting the matrix
        # add slack variable
        one_slack = np.zeros((self.t.tableau.shape[0],1), dtype=np.int)
        one_slack[-2] = 1
        self.t.tableau = np.c_[self.t.tableau[:,:-1], one_slack, self.t.tableau[:,-1][:,np.newaxis]]

        # A_m+1=A_m+1-A_(m+1,B)*A*
        A = self.t.matrix
        A_m1 = A[-1]
        A_m1B = A_m1[self.t.row_to_var]
        self.t.A_star[-1] -= A_m1B.dot(self.t.A_star[:-1])

        # S_m+1=-A_(m+1,B)*S*
        self.t.S_star[-1,:-1] = -A_m1B.dot(self.t.S_star[:-1,:-1])

        # b_m+1 = b_m+1-A_(m+1,B)b*
        self.t.bs[-1] -= A_m1B.dot(self.t.bs[:-1])

        # check if b is negative => not feasible => dual pivot
        self.t.row_to_var = np.append(self.t.row_to_var, self.t.tableau.shape[1]-2)
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
        self.t.constraints.append(constraint)
        frac_coefficients = constraint.x.get_coefficients(len(self.t.variables))
        for v_idx in range(len(frac_coefficients)):
            frac_coefficients[v_idx] = Fraction(frac_coefficients[v_idx])

        if self.t.dual:
            # add constraint as new variable
            coefficients = np.array(frac_coefficients)
            const_y = constraint.y
            if constraint.type != ">=":
                coefficients *= -1
                const_y *= -1
            self.add_new_col(coefficients, const_y)
        else:
            zeros = [0] * (self.t.tableau.shape[1] - 1 - len(self.t.variables))
            new_constraint_line = np.array(frac_coefficients + zeros + [constraint.y])
            if constraint.type != "<=":
                new_constraint_line *= -1
            new_constraint_line = new_constraint_line[np.newaxis, :]
            self.add_new_row(new_constraint_line)

    def to_dual(self):
        self.t.tableau = self.t.tableau.T
        # number of cols for variables changed =>
        self.t.nof_var_cols = self.t.tableau.shape[1]-1
        self.t.type = self.MAXIMIZE if self.t.type == self.MINIMIZE else self.MINIMIZE

    def to_standard(self):
        if self.t.type == self.MINIMIZE:
            self.t.tableau *= -1
        if np.any(self.t.bs < 0):
            # solve start problem
            first_obj = -np.copy(self.t.obj)
            # add x_0 variable
            ones = np.ones(self.t.matrix.shape[0]+1, dtype=np.int)
            self.t.tableau = np.c_[self.t.tableau[:, :-1], -ones, self.t.tableau[:, -1]]

            # and change obj to max -x_0
            self.t.tableau[-1,:] = np.zeros(self.t.tableau.shape[1], dtype=np.int)
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

                # update self.t.row_to_var
                self.t.row_to_var[self.t.row_to_var > self.t.nof_var_cols] -= 1

                rows = np.where(self.t.row_to_var < self.t.nof_var_cols)[0]
                vs = []
                for row in rows:
                    v = self.t.row_to_var[row]
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
        self.t.row_to_var = [False for x in range(self.t.matrix.shape[0])]
        # Build slack variables
        identity = np.eye(self.t.matrix.shape[0], dtype=np.int)
        identity = np.r_[identity, np.zeros((1,self.t.matrix.shape[0]), dtype=np.int)]
        self.t.tableau = np.c_[self.t.tableau[:,:-1], identity, self.t.tableau[:,-1]]

        # range not including the b column
        # get all columns which have only one value => basis
        row = 0
        for c in range(self.t.matrix.shape[1]-1-len(self.t.row_to_var),self.t.matrix.shape[1]-1):
            self.t.row_to_var[row] = c
            row += 1
        self.t.row_to_var = np.array(self.t.row_to_var)

        if self.p['start_conf']:
            print("Start Tableau:")
            print(self.t.tableau)


    def new_basis(self,entering,leaving):
        for row in range(self.t.matrix.shape[0]):
            if self.t.row_to_var[row] == leaving:
                self.t.row_to_var[row] = entering
                break

    def pivot(self):
        # check if the current tableau is optimal
        # if optimal every value in obj is non negative
        get_l_e = time()
        min_not_in_basis = np.copy(self.t.obj[:-1])
        min_not_in_basis[self.t.row_to_var] = 0 # non negative
        c = np.argmin(min_not_in_basis)
        get_l_e = time() - get_l_e
        if self.t.obj[c] < -0:
            positive = np.where(self.t.matrix[:,c] > 0)[0]
            if len(positive):
                entering = c
                l = np.argmin(self.t.bs[positive]/self.t.matrix[positive,c])
                leaving_row = positive[l]
                leaving = self.t.row_to_var[leaving_row]
            else:
                if self.t.dual:
                    raise InfeasibleError("The model is infeasible because the dual is unbound")
                else:
                    raise Unbounded(self.t.variables[c]["x"].name)
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
        leaving = self.t.row_to_var[leaving_row]
        self.new_basis(entering, leaving)

    def check_if_dual(self):
        if self.t.dual:
            return
        if self.t.type == self.MINIMIZE:
            if len(self.t.variables)/8 < len(self.t.constraints):
                self.t.dual = True

    def print_solution(self,slack=False):
        if self.t.dual:
            for c in range(self.t.nof_var_cols,self.t.nof_var_cols+len(self.t.variables)):
                v_idx = c-self.t.nof_var_cols
                if self.t.obj[c] > 0:
                    print("%s is %f" % (self.t.variables[v_idx]["x"].name, self.t.obj[c]))
        else:
            cor_to_variable = self.t.row_to_var < len(self.t.variables)
            for c in range(len(self.t.row_to_var)):
                if cor_to_variable[c]:
                    v_idx = self.t.row_to_var[c]
                    print("%s is %f" % (self.t.variables[v_idx]["x"].name,self.t.bs[c]))
        if slack:
            if self.t.dual:
                print("Problem solved using dual simplex => no slack variables possible")
            else:
                for c in range(len(self.t.row_to_var)):
                    if not cor_to_variable[c]:
                        v_idx = self.t.row_to_var[c]
                        print("slack %d is %f" % ((v_idx-len(self.t.variables)+1),self.t.bs[c]))
        if self.t.type == self.MAXIMIZE:
            print("Obj: %f" % (self.t.obj[-1]))
        elif self.t.type == self.MINIMIZE:
            print("Obj: %f" % (-self.t.obj[-1]))

    def print_constraints(self):
        for c in self.t.constraints:
            print(c)

    def get_solution_object(self):
        if self.t.dual:
            sol_row = list(self.t.obj[self.t.nof_var_cols:-1])
        else:
            cor_to_variable = self.t.row_to_var < len(self.t.variables)
            sol_row = [0]*len(self.t.variables)
            for c in range(len(self.t.row_to_var)):
                if cor_to_variable[c]:
                    v_idx = self.t.row_to_var[c]
                    sol_row[v_idx] = self.t.bs[c]
        if self.t.type == self.MAXIMIZE:
            sol_row.append(self.t.obj[-1])
        elif self.t.type == self.MINIMIZE:
            sol_row.append(-self.t.obj[-1])

        return sol_row

    def solve_from_scratch(self, consider_dual):
        if consider_dual is None:
            consider_dual = self.TEST_DUAL

        self.t.tableau = np.zeros((len(self.t.constraints)+1,len(self.t.variables)+1),dtype=object)

        if self.p['information']:
            print("Information: ")
            print("We have %d variables " % len(self.t.variables))
            print("We have %d constraints " % len(self.t.constraints))

        self.t.nof_var_cols = len(self.t.variables)
        i = 0
        for constraint in self.t.constraints:
            coefficients = constraint.x.get_coefficients(len(self.t.variables))
            # if constraint.y < 0:
            #     raise Unsolveable("All constants on the right side must be non-negative")
            values = coefficients + [constraint.y]
            for v_idx in range(len(values)):
                self.t.tableau[i][v_idx] = Fraction(values[v_idx])

            if self.t.type == self.MAXIMIZE:
                if constraint.type != "<=":
                    self.t.tableau[i] *= -1
            if self.t.type == self.MINIMIZE:
                if constraint.type != ">=":
                    self.t.tableau[i] *= -1
            i += 1

        # set obj
        for v_idx in range(len(self.t.obj_coefficients)):
            self.t.tableau[-1][v_idx] = self.t.obj_coefficients[v_idx]
        if consider_dual == self.DEF_DUAL:
            self.t.dual = True
        if consider_dual == self.TEST_DUAL:
            self.check_if_dual()
        if self.t.dual:
            if self.p['information']:
                print("Using dual")
            self.to_dual()

        self.t.tableau[-1, :] = -self.t.tableau[-1,:]

        self.to_standard()

        solved, _ = self.pivot()
        while not solved:
            s = time()
            solved, _ = self.pivot()
            self.steps += 1

    def create_from_tableau(self, ex_tv):
        self.t = ex_tv

    def solve_mip(self, bnbbp, level=0):
        # assume it's solved
        solved = True
        # check if the type of the solution is correct
        # maybe we need branch and bound for MIP
        # get "most real" var
        sol_arr = self.get_solution_object()[:-1]
        # print("sol_arr", sol_arr)
        i = 0
        max_diff = 0
        max_diff_i = 0
        for var in self.t.variables:
            if var['x'].type == "int+":
                if int(sol_arr[i]) != sol_arr[i]:
                    diff = abs(round(sol_arr[i]) - sol_arr[i])
                    # print(diff)
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_i = i
                    # print("%s should be int but is %s (~ %.2f)" %
                    #       (var['x'].name, sol_arr[i], sol_arr[i]))
                    solved = False
                else:
                    # print("%s is int and %d" % (var['x'].name, sol_arr[i]))
                    pass
            i += 1

        if not solved:
            # print("Isn't solved yet")
            # print(self.get_solution_object())
            # print("Branch and bound for var %s" % (self.t.variables[max_diff_i]['x'].name))
            bnb = BnB(bnbbp, self.t, self.t.variables[max_diff_i], sol_arr[max_diff_i],level+1)
            return False
        return True

    def solve(self, consider_dual=None,):
        self.solve_from_scratch(consider_dual)

        print("before mip", self.get_solution_object())
        # check if mip problem and solve if necessary
        bnbbp = BnBBP(self.t.obj_type, self)
        self.solve_mip(bnbbp)

        self.t = bnbbp.best_model.t

        if self.p['end_conf']:
            print("End tableau")
            print(np.around(self.t.tableau, decimals=4))
            print("Steps: ", self.steps)


def divInf(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [Inf, Inf, Inf] """
    # b[b==0] = np.finfo(np.float64).eps
    try:
        return a/b
    except ZeroDivisionError:
        result = []
        for i in range(len(a)):
            if b[i] == 0:
                result.append(np.inf)
            else:
                result.append(a[i]/b[i])
        return result
