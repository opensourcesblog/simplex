from Model.model import Model
import numpy as np
from time import time

# http://ftp.mcs.anl.gov/pub/tech_reports/reports/P602.pdf

m = Model(print_obj={
   'start_conf': True
})

a = m.add_var("real+", name="corn")
b = m.add_var("real+", name="milk")
c = m.add_var("real+", name="bread")


m.minimize(0.18*a+0.23*b+0.05*c)

# vitamin a
m.add_constraint(107*a+500*b+0*c >= 5000)
# cal
m.add_constraint(72*a+121*b+65*c >= 2000)



t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution()
