from Model.model import Model
import numpy as np
from time import time

m = Model()

"""
A post office example:
Minimize the number of full time employees
Fulltime = 5 consecutive days + 2 free days


Weekday   | Employees needed
Monday    | 17
Tuesday   | 13
Wednesday | 15
Thursday  | 19
Friday    | 14
Saturday  | 16
Sunday    | 11

"""


mon = m.add_var("real+", name="Monday")
tue = m.add_var("real+", name="Tuesday")
wed = m.add_var("real+", name="Wednesday")
thu = m.add_var("real+", name="Thursday")
fri = m.add_var("real+", name="Friday")
sat = m.add_var("real+", name="Saturday")
sun = m.add_var("real+", name="Sunday")


m.minimize(mon+tue+wed+thu+fri+sat+sun)

m.add_constraint(mon+thu+fri+sat+sun >= 17)
m.add_constraint(tue+fri+sat+sun+mon >= 13)
m.add_constraint(wed+sat+sun+mon+tue >= 15)
m.add_constraint(thu+sun+mon+tue+wed >= 19)
m.add_constraint(fri+mon+tue+wed+thu >= 14)
m.add_constraint(sat+tue+wed+thu+fri >= 16)
m.add_constraint(sun+wed+thu+fri+sat >= 11)



t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution()
