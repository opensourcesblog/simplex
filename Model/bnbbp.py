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
