from Model.model import Model
import numpy as np
from time import time

m = Model()

"""
A Manufacturing Example

Machine | Product P | Product Q | Product R | Availability
A       | 20        | 10        | 10        | 2400
B       | 12        | 28        | 16        | 2400
C       | 15        |  6        | 16        | 2400
D       | 10        | 15        |  0        | 2400
Total   | 57        | 59        | 42        | 9600


Item          | Product P | Product Q | Product R
Revenue p.u.  | 90$       | 100$      | 70$
Material p.u. | 45$       | 40$       | 20$
Profit p.u.   | 45$       | 60$       | 50$
Maximum sales | 100       | 40        | 60

Maximize profit
"""


p = m.add_var("real+", name="p")
q = m.add_var("real+", name="q")
r = m.add_var("real+", name="r")


m.maximize(45*p+60*q+50*r)

m.add_constraint(20*p+10*q+10*r <= 2400)
m.add_constraint(12*p+28*q+16*r <= 2400)
m.add_constraint(15*p+6*q+16*r <= 2400)
m.add_constraint(10*p+15*q <= 2400)
m.add_constraint(p <= 100)
m.add_constraint(q <= 40)
m.add_constraint(r <= 60)



t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution()
