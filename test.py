import unittest
from inspect import ismethod
from Model.model import Model
import numpy as np
from Model.error import *


class MyTest(unittest.TestCase):
    def test_maximize_2v_4c_1o(self):

        m = Model()

        x = []
        for i in range(1,3):
            x.append(m.add_var("real+", name="x_%d" % i))


        m.maximize(3*x[1])


        m.add_constraint(2*x[0]+3*x[1]  <= 6)
        m.add_constraint(-3*x[0]+2*x[1] <= 3)
        m.add_constraint(0*x[0]+2*x[1]  <= 5)
        m.add_constraint(2*x[0]+1*x[1]  <= 4)

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [0.23076923076923078, 1.8461538461538463, 0, 0, 1.3076923076923077, 1.6923076923076923, 5.5384615384615383]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_maximize_2v_4c_2o(self):
        m = Model()

        x = []
        for i in range(1,3):
            x.append(m.add_var("real+", name="x_%d" % i))


        m.maximize(4*x[0]+3*x[1])


        m.add_constraint(2*x[0]+3*x[1]  <= 6)
        m.add_constraint(-3*x[0]+2*x[1] <= 3)
        m.add_constraint(0*x[0]+2*x[1]  <= 5)
        m.add_constraint(2*x[0]+1*x[1]  <= 4)

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [1.5, 1, 0, 5.5, 3.0, 0, 9]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_maximize_4v_3c_3o(self):
        m = Model()

        x = []
        for i in range(1,5):
            x.append(m.add_var("real+", name="x_%d" % i))


        m.maximize(3*x[1]+x[2]+4*x[3])

        m.add_constraint(x[0]+x[1]+x[2]+x[3] <= 40)
        m.add_constraint(2*x[0]+x[1]+(-1*x[2])+(-1*x[3]) <= 10)
        m.add_constraint(x[3]+(-1*x[1]) <= 10)

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [0, 15.0, 0, 25.0, 0, 20.0, 0, 145.0]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_minimize_2v_3c_2o(self):
        m = Model()

        x = []
        for i in range(1,3):
            x.append(m.add_var("real+", name="x_%d" % i))

        m.minimize(0.12*x[0]+0.15*x[1])

        m.add_constraint(60*x[0]+60*x[1] >= 300)
        m.add_constraint(12*x[0]+ 6*x[1] >= 36)
        m.add_constraint(10*x[0]+30*x[1] >= 90)

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [3, 2, 0, 12, 0, 0.66]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_unbound(self):
        m = Model()

        a = m.add_var("real+", name="a")
        b = m.add_var("real+", name="b")


        m.maximize(3*a-b)

        m.add_constraint(-3*a+3*b <= 6)
        m.add_constraint(-8*a+4*b <= 4)

        try:
            m.solve()
        except Unbounded as e:
            pass
        else:
            self.fail("Should raise Unbounded but didn't")


if __name__ == '__main__':
    unittest.main()
    # t = MyTest()
    # t.test_minimize_2v_3c_2o()
