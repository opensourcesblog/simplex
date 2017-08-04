from Model.model import Model
import numpy as np
from time import time
import sys

m = Model(print_obj={
   'start_conf': True
})

a = m.add_var("real+", name="a")
b = m.add_var("real+", name="b")


m.maximize(3*a-b)

m.add_constraint(-3*a+3*b <= 6)
m.add_constraint(-8*a+4*b <= 4)



t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution()
