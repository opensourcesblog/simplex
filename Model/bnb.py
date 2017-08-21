import numpy as np
import math
from .error import *


class BnB:
    def __init__(self, bnbbp, t, var, value, level=0):
        self.relaxedTV_lower = t.deepcopy()
        self.relaxedTV_upper = t.deepcopy()
        self.level = level
        self.bnbbp = bnbbp

        lower_feasible, lower_solved, lower_model = self.solve_lower(var, value)
        if lower_solved:
            sol = lower_model.get_solution_object()
            if bnbbp.check_if_better(sol[-1],lower_model):
                print("Found better", sol[-1])
            echo(sol, level)
        elif lower_feasible:
            echo(lower_model.get_solution_object(), level)

        upper_feasible, upper_solved, upper_model = self.solve_upper(var, value)
        if upper_solved:
            sol = upper_model.get_solution_object()
            if bnbbp.check_if_better(sol[-1], upper_model):
                print("Found better", sol[-1])
            echo(sol, level)
        elif upper_feasible:
            echo(upper_model.get_solution_object(), level)

    def solve_lower(self, var, value):
        from .model import Model

        lower_value = math.floor(value)

        m_lower = Model()
        m_lower.create_from_tableau(ex_tv=self.relaxedTV_lower)
        try:
            m_lower.add_lazy_constraint(var['x'] <= lower_value)
            if not self.bnbbp.check_if_bound(m_lower.get_solution_object()[-1]):
                solved_lower = m_lower.solve_mip(self.bnbbp, self.level)
                if solved_lower:
                    return True, True, m_lower
            return True, False, m_lower
        except InfeasibleError:
            return False, False, m_lower

    def solve_upper(self, var, value):
        from .model import Model

        upper_value = math.ceil(value)

        m_upper = Model()
        m_upper.create_from_tableau(ex_tv=self.relaxedTV_upper)
        try:
            m_upper.add_lazy_constraint(var['x'] >= upper_value)
            if not self.bnbbp.check_if_bound(m_upper.get_solution_object()[-1]):
                solved_upper = m_upper.solve_mip(self.bnbbp, self.level)
                if solved_upper:
                    return True, True, m_upper
            return True, False, m_upper
        except InfeasibleError:
            return False, False, m_upper


def echo(val, level):
    s = " "*(level*2)
    # print(s+str(val))
