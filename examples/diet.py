import numpy as np
from time import time

from Model.model import Model

m = Model(print_obj={
   'start_conf': True
})

a = m.add_var("real+", name="oat")
b = m.add_var("real+", name="chicken")
c = m.add_var("real+", name="egg")
d = m.add_var("real+", name="milk")
e = m.add_var("real+", name="cake")
f = m.add_var("real+", name="bean")


m.minimize(25*a+130*b+85*c+70*d+95*e+98*f)

# calories
m.add_constraint(110*a+205*b+160*c+160*d+420*e+260*f >= 2000)
# proteins
m.add_constraint(4*a+32*b+13*c+8*d+4*e+14*f >= 55)
# calcium
m.add_constraint(2*a+12*b+54*c+285*d+22*e+80*f >= 800)

# oats
m.add_constraint(a <= 4)
# chicken
m.add_constraint(b <= 3)
# egg
m.add_constraint(c <= 2)
# milk
m.add_constraint(d <= 8)
# cake
m.add_constraint(e <= 1)
# bean
m.add_constraint(f <= 2)


t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution()
