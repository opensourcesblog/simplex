import numpy as np


class BnBBP:
    def __init__(self, type, best_model=False):
        self.MINIMIZE = -1
        self.MAXIMIZE = 1
        if type == self.MAXIMIZE:
            self.best_obj = -np.inf
        else:
            self.best_obj = np.inf
        self.best_model = best_model
        self.type = type

    def check_if_better(self, val, model):
        if self.type == self.MAXIMIZE:
            if val > self.best_obj:
                self.best_obj = val
                self.best_model = model
                return True
        elif self.type == self.MINIMIZE:
            if val < self.best_obj:
                self.best_obj = val
                self.best_model = model
                return True
        return False

    def check_if_bound(self, val):
        if self.type == self.MAXIMIZE:
            if val < self.best_obj:
                return True
        elif self.type == self.MINIMIZE:
            if val > self.best_obj:
                return True
        return False

    def get_better_and_worse(self, m1, m2):
        if m1.is_infeasible:
            obj_m2 = m2.get_solution_object()[-1]
            if self.check_if_bound(obj_m2):
                return False, False
            else:
                return m2, False
        if m2.is_infeasible:
            obj_m1 = m1.get_solution_object()[-1]
            if self.check_if_bound(obj_m1):
                return False, False
            else:
                return m1, False

        obj_m1 = m1.get_solution_object()[-1]
        obj_m2 = m2.get_solution_object()[-1]

        if self.check_if_bound(obj_m1):
            m1_n = False
        else:
            m1_n = m1
        if self.check_if_bound(obj_m2):
            m2_n = False
        else:
            m2_n = m2

        if self.type == self.MAXIMIZE:
            if obj_m1 > obj_m2:
                return m1_n, m2_n
            else:
                return m2_n, m1_n
        if self.type == self.MINIMIZE:
            if obj_m1 < obj_m2:
                return m1_n, m2_n
            else:
                return m2_n, m1_n
