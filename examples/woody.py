from Model.model import Model
import numpy as np
from time import time
import sys

m = Model(print_obj={
    'start_conf': True
})

x1 = m.add_var("real+", name="a")
x2 = m.add_var("real+", name="b")
x3 = m.add_var("real+", name="c")

m.minimize(35 * x1 + 60 * x2 + 75 * x3)

# """
m.add_constraint(8 * x1 + 12 * x2 + 16 * x3 >= 120)
m.add_constraint(15 * x2 + 20 * x3 >= 60)
m.add_constraint(3 * x1 + 6 * x2 + 9 * x3 >= 48)
"""
m.add_constraint(8 * x1 + 12 * x2 + 16 * x3 + 12 * x4 <= 120)
m.add_constraint(15 * x2 + 20 * x3 + 30 * x4 <= 60)
m.add_constraint(3 * x1 + 6 * x2 + 9 * x3 + 15 * x4 <= 48)
"""

m.solve(consider_dual=0)
m.print_solution()
print(m.t.tableau)
m.add_new_variable([12, 30, 10], 15)

m.print_solution()
