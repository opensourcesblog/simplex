import numpy as np


class Tableau(object):
    def __init__(self):
        self._data = np.zeros((1,1))
        self._nof_var_cols = 0

    def __set__(self, obj, val):
        self._data = val

    def __get__(self, obj, obj_type):
        return self._data

    def get_obj(self,tv):
        return self._data[-1]

    def set_obj(self, tv, value):
        self._data[-1] = value

    def get_nof_var_cols(self,tv):
        return self._nof_var_cols

    def set_nof_var_cols(self, tv, value):
        self._nof_var_cols = value

    def get_bs(self,tv):
        return self._data[:-1,-1]

    def set_bs(self, tv, value):
        self._data[:-1,-1] = value

    def get_matrix(self,tv):
        return self._data[:-1]

    def set_matrix(self, tv, value):
        self._data[:-1] = value

    def get_A_star(self, tv):
        return self._data[:-1, :self._nof_var_cols]

    def set_A_star(self, tv, value):
        self._data[:-1, :self._nof_var_cols] = value

    def get_S_star(self, tv):
        return self._data[:-1, self._nof_var_cols:-1]

    def set_S_star(self, tv, value):
        self._data[:-1, self._nof_var_cols:-1] = value

    def get_y_star(self, tv):
        return self._data[-1, self._nof_var_cols:-1]

    def set_y_star(self, tv, value):
        self._data[-1, self._nof_var_cols:-1] = value

    def get_c_star(self, tv):
        return self._data[-1, :self._nof_var_cols]

    def set_c_star(self, tv, value):
        self._data[-1, :self._nof_var_cols] = value


class TableauView(object):
    tableau = Tableau()
    obj = property(tableau.get_obj,tableau.set_obj)
    nof_var_cols = property(tableau.get_nof_var_cols, tableau.set_nof_var_cols)
    bs = property(tableau.get_bs, tableau.set_bs)
    matrix = property(tableau.get_matrix, tableau.set_matrix)
    A_star = property(tableau.get_A_star, tableau.set_A_star)
    S_star = property(tableau.get_S_star, tableau.set_S_star)
    y_star = property(tableau.get_y_star, tableau.set_y_star)
    c_star = property(tableau.get_c_star, tableau.set_c_star)






