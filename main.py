from Model.model import Model

def test(cons):
    print("Constraint:")
    print(cons)
    print("X:", cons.x)
    print("Y:", cons.y)
    feasible = cons.check_feasibility()
    if feasible:
        print("Still feasible")
    else:
        print("Not feasible")

m = Model()

x = []
for i in range(5):
    x.append(m.add_var("real+", name="x_%d" % i))

m.minimize(sum(x))

"""
3 2 1 0 0 = 1
5 1 0 1 0 = 3
2 5 0 0 1 = 4

Solution:
0 0.5 0 2.5 1.5
"""


m.add_constraint(3*x[0]+2*x[1]+1*x[2] == 1)
m.add_constraint(5*x[0]+1*x[1]        +1*x[3] == 3)
m.add_constraint(2*x[0]+5*x[1]               +1*x[4] == 4)

m.solve()
m.print_solution()
