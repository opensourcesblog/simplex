import numpy as np
from time import time
import random, string
from fractions import Fraction

random.seed(9001)

from Model.model import Model

m = Model(print_obj={
   'start_conf': True,
   'end_conf': True
})

x = []
for i in range(3):
    x.append(m.add_var("real+", name=i))
x = np.array(x)

m.maximize(sum(np.array([5,5,3])*x))


m.add_constraint(x[0]+3*x[1]+x[2] <= 3)
m.add_constraint(-x[0]+0*x[1]+3*x[2] <= 2)
m.add_constraint(2*x[0]-x[1]+2*x[2] <= 4)
m.add_constraint(2*x[0]+3*x[1]-x[2] <= 2)

print("all added")

t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution(slack=False)
print("Steps: ",m.steps)

