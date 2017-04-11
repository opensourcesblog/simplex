from Model.model import Model
import numpy as np
from time import time

m = Model()

x = []
for i in range(1,5):
    x.append(m.add_var("real+", name="x_%d" % i))


m.maximize(3*x[1]+x[2]+4*x[3])

m.add_constraint(x[0]+x[1]+x[2]+x[3] <= 40)
m.add_constraint(2*x[0]+x[1]-x[2]-x[3] <= 10)
m.add_constraint(x[3]-x[1] <= 10)


t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))
m.print_solution()
