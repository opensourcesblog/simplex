import numpy as np
import copy

class Tableau(object):
    def __init__(self, data=False, nof_var_cols=0, dual=False, obj_type=1, type=1,
                 row_to_var= False, variables= False, constraints=False, lazy_constraints= False,
                 obj_coefficients=False):
        self.MINIMIZE = -1
        self.MAXIMIZE = 1

        if data is False:
            data = np.zeros((1, 1))
        self._data = data
        self._nof_var_cols = nof_var_cols
        self._dual = dual
        self._obj_type = obj_type
        self._type = type
        if row_to_var is False:
            row_to_var = np.array([])
        self._row_to_var = row_to_var
        self._variables = variables
        self._constraints = constraints
        self._lazy_constraints = lazy_constraints
        if obj_coefficients is False:
            obj_coefficients = np.array([])
        self._obj_coefficients = obj_coefficients

    def set_tableau(self, val):
        self._data = val

    def get_tableau(self):
        return self._data

    def get_obj(self):
        return self._data[-1]

    def set_obj(self, value):
        self._data[-1] = value

    def get_nof_var_cols(self):
        return self._nof_var_cols

    def set_nof_var_cols(self, value):
        self._nof_var_cols = value

    def get_bs(self):
        return self._data[:-1,-1]

    def set_bs(self, value):
        self._data[:-1,-1] = value

    def get_matrix(self):
        return self._data[:-1]

    def set_matrix(self, value):
        self._data[:-1] = value

    def get_A_star(self):
        return self._data[:-1, :self._nof_var_cols]

    def set_A_star(self, value):
        self._data[:-1, :self._nof_var_cols] = value

    def get_S_star(self):
        return self._data[:-1, self._nof_var_cols:-1]

    def set_S_star(self, value):
        self._data[:-1, self._nof_var_cols:-1] = value

    def get_y_star(self):
        return self._data[-1, self._nof_var_cols:-1]

    def set_y_star(self, value):
        self._data[-1, self._nof_var_cols:-1] = value

    def get_c_star(self):
        return self._data[-1, :self._nof_var_cols]

    def set_c_star(self, value):
        self._data[-1, :self._nof_var_cols] = value

    def get_dual(self):
        return self._dual

    def set_dual(self, value):
        self._dual = value

    def get_obj_type(self):
        return self._obj_type

    def set_obj_type(self, value):
        self._obj_type = value

    def get_type(self):
        return self._type

    def set_type(self, value):
        self._type = value

    def get_row_to_var(self):
        return self._row_to_var

    def set_row_to_var(self, value):
        self._row_to_var = value

    def get_variables(self):
        return self._variables

    def set_variables(self, value):
        self._variables = value

    def get_constraints(self):
        return self._constraints

    def set_constraints(self, value):
        self._constraints = value

    def get_lazy_constraints(self):
        return self._lazy_constraints

    def set_lazy_constraints(self, value):
        self._lazy_constraints = value

    def get_obj_coefficients(self):
        return self._obj_coefficients

    def set_obj_coefficients(self, value):
        self._obj_coefficients = value

    def get_basis(self):
        return self._data[:-1,:-1][:,self._row_to_var]

    def get_non_basis(self):
        non_row_to_var = np.ones(self._data.shape[1]-1, np.bool)
        non_row_to_var[self._row_to_var] = 0
        return self._data[:-1,:-1][:,non_row_to_var]

    def get_non_basis_c(self):
        non_row_to_var = np.ones(self._data.shape[1]-1, np.bool)
        non_row_to_var[self._row_to_var] = 0
        return self._data[-1,:-1][non_row_to_var]

    def get_c_b(self):
        return -self._data[-1][self._row_to_var]

    def get_non_basis_cols(self):
        non_basis_columns = np.ones(self._data.shape[1] - 1)
        non_basis_columns[self._row_to_var] = 0
        return non_basis_columns.nonzero()[0]

    def get_A(self):
        return self._data[:-1,:-1]

    def frac_print(self):
        # compute len for each element
        shape = self._data.shape if self._data.ndim == 2 else (1, self._data.shape[0])

        len_arr = [[0] * shape[1] for x in range(shape[0])]
        str_arr = [[""] * shape[1] for x in range(shape[0])]
        # str_arr =
        for y in range(shape[0]):
            for x in range(shape[1]):
                n_val = self._data[y, x].numerator if self._data.ndim == 2 else self._data[x].numerator
                d_val = self._data[y, x].denominator if self._data.ndim == 2 else self._data[x].denominator
                frac = n_val / d_val
                if int(frac) == frac:
                    len_arr[y][x] = len(" %d" % frac)
                    str_arr[y][x] = " %d" % frac
                else:
                    len_arr[y][x] = len(" %d/%d" % (n_val, d_val))
                    str_arr[y][x] = " %d/%d" % (n_val, d_val)

        str_arr_col = list(zip(*str_arr))
        i = 0
        for col in zip(*len_arr):
            max_val = max(col) + 2
            str_arr_col[i] = [" " * (max_val - len(x)) + str(x) for x in str_arr_col[i]]
            i += 1

        str_arr = list(zip(*str_arr_col))
        output = ""
        for row in str_arr:
            line = "["
            for ele in row:
                line += ele
            output += line + "]\n"

        return output[:-1]

    def float_print(self):
        # compute len for each element
        shape = self._data.shape if self._data.ndim == 2 else (1, self._data.shape[0])

        len_arr = [[0] * shape[1] for x in range(shape[0])]
        str_arr = [[""] * shape[1] for x in range(shape[0])]
        # str_arr =
        for y in range(shape[0]):
            for x in range(shape[1]):
                n_val = self._data[y, x].numerator if self._data.ndim == 2 else self._data[x].numerator
                d_val = self._data[y, x].denominator if self._data.ndim == 2 else self._data[x].denominator
                frac = n_val / d_val
                if int(frac) == frac:
                    len_arr[y][x] = len(" %d" % frac)
                    str_arr[y][x] = " %d" % frac
                else:
                    len_arr[y][x] = len(" %.2f" % (n_val/d_val))
                    str_arr[y][x] = " %.2f" % (n_val/d_val)

        str_arr_col = list(zip(*str_arr))
        i = 0
        for col in zip(*len_arr):
            max_val = max(col) + 2
            str_arr_col[i] = [" " * (max_val - len(x)) + str(x) for x in str_arr_col[i]]
            i += 1

        str_arr = list(zip(*str_arr_col))
        output = ""
        for row in str_arr:
            line = "["
            for ele in row:
                line += ele
            output += line + "]\n"

        return output[:-1]

class TableauView(object):
    def __init__(self,tab_data=False):
        if tab_data is False:
            self.tab = Tableau()
        else:
            self.tab = tab_data

    def deepcopy(self):
        tableau = np.copy(self.tab.get_tableau())
        nof_var_cols = self.tab.get_nof_var_cols()
        dual = self.tab.get_dual()
        type = self.tab.get_type()
        obj_type = self.tab.get_obj_type()
        row_to_var = np.copy(self.tab.get_row_to_var())
        variables = copy.deepcopy(self.tab.get_variables())
        constraints = copy.deepcopy(self.tab.get_constraints())
        lazy_constraints = copy.deepcopy(self.tab.get_lazy_constraints())
        obj_coefficients = copy.deepcopy(self.tab.get_obj_coefficients())
        tab = Tableau(tableau, nof_var_cols, dual, obj_type, type, row_to_var,
                      variables, constraints, lazy_constraints, obj_coefficients)
        return TableauView(tab)

    @property
    def tableau(self):
        return self.tab.get_tableau()

    @tableau.setter
    def tableau(self, data):
        self.tab.set_tableau(data)

    @property
    def matrix(self):
        return self.tab.get_matrix()

    @matrix.setter
    def matrix(self, data):
        self.tab.set_matrix(data)

    @property
    def nof_var_cols(self):
        return self.tab.get_nof_var_cols()

    @nof_var_cols.setter
    def nof_var_cols(self, data):
        self.tab.set_nof_var_cols(data)

    @property
    def A_star(self):
        return self.tab.get_A_star()

    @A_star.setter
    def A_star(self, data):
        self.tab.set_A_star(data)

    @property
    def bs(self):
        return self.tab.get_bs()

    @bs.setter
    def bs(self, data):
        self.tab.set_bs(data)

    @property
    def obj(self):
        return self.tab.get_obj()

    @obj.setter
    def obj(self, data):
        self.tab.set_obj(data)

    @property
    def S_star(self):
        return self.tab.get_S_star()

    @S_star.setter
    def S_star(self, data):
        self.tab.set_S_star(data)

    @property
    def y_star(self):
        return self.tab.get_y_star()

    @y_star.setter
    def y_star(self, data):
        self.tab.set_y_star(data)

    @property
    def c_star(self):
        return self.tab.get_c_star()

    @c_star.setter
    def c_star(self, data):
        self.tab.set_c_star(data)

    @property
    def dual(self):
        return self.tab.get_dual()

    @dual.setter
    def dual(self, data):
        self.tab.set_dual(data)

    @property
    def type(self):
        return self.tab.get_type()

    @type.setter
    def type(self, data):
        self.tab.set_type(data)

    @property
    def row_to_var(self):
        return self.tab.get_row_to_var()

    @row_to_var.setter
    def row_to_var(self, data):
        self.tab.set_row_to_var(data)

    @property
    def variables(self):
        return self.tab.get_variables()

    @variables.setter
    def variables(self, data):
        self.tab.set_variables(data)

    @property
    def constraints(self):
        return self.tab.get_constraints()

    @constraints.setter
    def constraints(self, data):
        self.tab.set_constraints(data)

    @property
    def lazy_constraints(self):
        return self.tab.get_lazy_constraints()

    @lazy_constraints.setter
    def lazy_constraints(self, data):
        self.tab.set_lazy_constraints(data)

    @property
    def obj_coefficients(self):
        return self.tab.get_obj_coefficients()

    @obj_coefficients.setter
    def obj_coefficients(self, data):
        self.tab.set_obj_coefficients(data)

    @property
    def obj_type(self):
        return self.tab.get_obj_type()

    @obj_type.setter
    def obj_type(self, data):
        self.tab.set_obj_type(data)

    @property
    def basis(self):
        return self.tab.get_basis()

    @property
    def non_basis(self):
        return self.tab.get_non_basis()

    @property
    def c_b(self):
        return self.tab.get_c_b()

    @property
    def c_n(self):
        return self.tab.get_non_basis_c()

    @property
    def cols_nb(self):
        return self.tab.get_non_basis_cols()

    @property
    def A(self):
        return self.tab.get_A()








