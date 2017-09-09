import numpy as np
import math
from .error import *
from binarytree import tree, convert, pprint


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


class BnB:
    def __init__(self, type, best_model=False):
        self.MINIMIZE = -1
        self.MAXIMIZE = 1
        if type == self.MAXIMIZE:
            self.best_obj = -np.inf
        else:
            self.best_obj = np.inf
        self.best_model = best_model
        self.type = type
        self.tree_model = [best_model]
        self.tree_obj = [best_model.get_solution_object()[-1]]

    def extend_tree(self, idx):
        self.tree_model += [None for x in range(len(self.tree_model), idx + 1)]
        if self.type == self.MAXIMIZE:
            self.tree_obj.extend([-np.inf for x in range(len(self.tree_obj), idx + 1)])
        else:
            self.tree_obj.extend([np.inf for x in range(len(self.tree_obj), idx + 1)])

    def get_obj_or_inf(self, model):
        if model.is_infeasible:
            if self.type == self.MAXIMIZE:
                return -np.inf
            else:
                return np.inf
        return model.get_solution_object()[-1]

    def add_left(self, last_idx, model):
        idx = last_idx * 2 + 1
        self.extend_tree(idx)

        self.tree_model[idx] = model
        self.tree_obj[idx] = self.get_obj_or_inf(model)
        return idx

    def add_right(self, last_idx, model):
        idx = last_idx * 2 + 2
        self.extend_tree(idx)

        self.tree_model[idx] = model
        self.tree_obj[idx] = self.get_obj_or_inf(model)
        return idx

    def remove_from_tree(self, idx):
        self.tree_model[idx] = None
        if self.type == self.MAXIMIZE:
            self.tree_obj[idx] = -np.inf
        else:
            self.tree_obj[idx] = np.inf

    def get_best_unexplored(self):
        if self.type == self.MAXIMIZE:
            idx = argmax(self.tree_obj)
            return self.tree_model[idx], idx
        else:
            idx = argmin(self.tree_obj)
            return self.tree_model[idx], idx

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

    def add(self, t, var, value, dtype, index=0):
        def solve_lower(var, value):
            from .model import Model

            lower_value = math.floor(value)

            m_lower = Model(dtype=dtype)
            m_lower.create_from_tableau(ex_tv=relaxedTV_lower)
            try:
                m_lower.add_lazy_constraint(var['x'] <= lower_value)
                return True, m_lower
            except InfeasibleError:
                return False, m_lower

        def solve_upper(var, value):
            from .model import Model

            upper_value = math.ceil(value)

            m_upper = Model(dtype=dtype)
            m_upper.create_from_tableau(ex_tv=relaxedTV_upper)
            try:
                m_upper.add_lazy_constraint(var['x'] >= upper_value)
                return True, m_upper
            except InfeasibleError:
                return False, m_upper

        def to_string(s):
            if isinstance(s, float):
                if np.isinf(s):
                    return ''
                else:
                    return str(s)
            else:
                return str(s)

        print("Branch and bound index", index)
        relaxedTV_lower = t.deepcopy()
        relaxedTV_upper = t.deepcopy()

        self.remove_from_tree(index)

        lower_feasible, lower_model = solve_lower(var, value)
        upper_feasible, upper_model = solve_upper(var, value)
        print("Solved lower and upper")

        self.add_left(index, lower_model)
        self.add_right(index, upper_model)

        obj_list = [to_string(x) for x in self.tree_obj]
        tree = convert(obj_list)
        pprint(tree)

        best_unexplored_model, best_index = self.get_best_unexplored()
        print("Best index", best_index)

        if best_unexplored_model:
            solved = best_unexplored_model.solve_mip(self, best_index)
            if solved:
                obj = best_unexplored_model.get_solution_object()[-1]
                if self.check_if_better(obj, best_unexplored_model):
                    print("Found best", obj)
