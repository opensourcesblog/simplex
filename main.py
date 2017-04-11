from Model.model import Model
import numpy as np
from time import time

m = Model(print_obj={
   'start_conf': True
})

x = []
for i in range(1,3):
    x.append(m.add_var("real+", name="x_%d" % i))


m.minimize(0.12*x[0]+0.15*x[1])

m.add_constraint(60*x[0]+60*x[1] >= 300)
m.add_constraint(12*x[0]+ 6*x[1] >= 36)
m.add_constraint(10*x[0]+30*x[1] >= 90)


t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))



m.print_solution()
