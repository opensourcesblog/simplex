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
for i in range(1,6):
    x.append(m.add_var("real+", name="x_%d" % i))

m.minimize(sum(x))

m.add_constraint(3*x[0]+2*x[1]+1*x[2] == 1)
m.add_constraint(5*x[0]+1*x[1]+1*x[2]+1*x[3] == 3)
m.add_constraint(2*x[0]+5*x[1]+1*x[2]        +1*x[4] == 4)

m.solve()
