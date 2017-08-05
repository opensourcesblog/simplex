from Model.model import Model
import numpy as np
from time import time

# http://ftp.mcs.anl.gov/pub/tech_reports/reports/P602.pdf

m = Model(print_obj={
   'start_conf': True
})

x1 = m.add_var("real+", name="x1")
x2 = m.add_var("real+", name="x2")
x3 = m.add_var("real+", name="x3")


m.maximize(x1+(-1*x2)+x3)

m.add_constraint(2*x1+(-1)*x2+2*x3 <= 4)
m.add_constraint(2*x1+(-3)*x2+x3 <= -5)
m.add_constraint(-1*x1+x2+(-2*x3) <= -1)




t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution()
