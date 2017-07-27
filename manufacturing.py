from Model.model import Model
import numpy as np
from time import time


m = Model(print_obj={
   'start_conf': True
})

"""
A Manufacturing Example

Machine | Product A | Product B | Product C | Availability
I       | 20        | 10        | 10        | 2400
J       | 12        | 28        | 16        | 2400
K       | 15        |  6        | 16        | 2400
L       | 10        | 15        |  0        | 2400
Total   | 57        | 59        | 42        | 9600


Item          | Product A | Product B | Product C
Revenue p.u.  | 90$       | 100$      | 70$
Material p.u. | 45$       | 40$       | 20$
Profit p.u.   | 45$       | 60$       | 50$
Maximum sales | 100       | 40        | 60

Maximize profit
"""


a = m.add_var("real+", name="apple")
b = m.add_var("real+", name="banana")
c = m.add_var("real+", name="carrot")


m.maximize(45*a+60*b+50*c)

m.add_constraint(20*a+10*b+10*c <= 2400)
m.add_constraint(12*a+28*b+16*c <= 2400)
m.add_constraint(15*a+6*b+16*c <= 2400)
m.add_constraint(10*a+15*b <= 2400)
m.add_constraint(a <= 100)
m.add_constraint(b <= 40)
m.add_constraint(c <= 60)



t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution()
