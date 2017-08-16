import numpy as np
from time import time
import random, string

from Model.model import Model

m = Model(print_obj={
   'end_conf': True
})

x = []
for i in range(4):
    x.append(m.add_var("real+", name=i))
x = np.array(x)

m.maximize(sum(np.array([4,1,5,3])*x))


m.add_constraint(x[0]-x[1]-x[2]+3*x[3] <= 1)
m.add_constraint(5*x[0]+x[1]+3*x[2]+8*x[3] <= 55)
m.add_constraint(-x[0]+2*x[1]+3*x[2]-5*x[3] <= 3)

print("all added")

t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution(slack=False)
print("Steps: ",m.steps)
