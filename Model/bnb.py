import numpy as np
import math
from .error import *


class BnB:
    def __init__(self, bnbbp, t, var, value, level=0):
        print("Branch and bound level", level)
        self.relaxedTV_lower = t.deepcopy()
        self.relaxedTV_upper = t.deepcopy()
        self.level = level
        self.bnbbp = bnbbp

        lower_feasible, lower_model = self.solve_lower(var, value)
        print("lower solved")
        upper_feasible, upper_model = self.solve_upper(var, value)
        print("upper solved")


        # returns False if infeasible or bounded
        better_model, worse_model = bnbbp.get_better_and_worse(lower_model, upper_model)

        if better_model:
            solved_better = better_model.solve_mip(self.bnbbp, self.level)
            if solved_better:
                obj = better_model.get_solution_object()[-1]
                if bnbbp.check_if_better(obj, better_model):
                    print("Found better", obj)
                    return

        if worse_model:
            solved_worse = worse_model.solve_mip(self.bnbbp, self.level)
            if solved_worse:
                obj = worse_model.get_solution_object()[-1]
                if bnbbp.check_if_better(obj, worse_model):
                    print("Found better", obj)

    def solve_lower(self, var, value):
        from .model import Model

        lower_value = math.floor(value)

        m_lower = Model()
        m_lower.create_from_tableau(ex_tv=self.relaxedTV_lower)
        print("created from tableau")
        try:
            m_lower.add_lazy_constraint(var['x'] <= lower_value)
            return True, m_lower
        except InfeasibleError:
            return False, m_lower

    def solve_upper(self, var, value):
        from .model import Model

        upper_value = math.ceil(value)

        m_upper = Model()
        m_upper.create_from_tableau(ex_tv=self.relaxedTV_upper)
        print("created from tableau")
        try:
            m_upper.add_lazy_constraint(var['x'] >= upper_value)
            return True, m_upper
        except InfeasibleError:
            return False, m_upper



