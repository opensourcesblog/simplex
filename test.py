import unittest
from inspect import ismethod
from Model.model import Model
import numpy as np
from Model.error import *
import random, string
import pickle
from time import time

def save_obj(obj, name ):
    with open('test_obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open('test_obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_by_key(obj, key, key_list=False):
    arr = []
    if key_list:
        for ing in key_list:
            arr.append(obj[ing][key])
    else:
        for ing in obj:
            arr.append(obj[ing][key])
    return np.array(arr)


class MyTest(unittest.TestCase):
    def test_maximize_2v_4c_1o(self):

        m = Model(print_obj={})

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
        real_sol = [0.23076923076923078, 1.8461538461538463, 5.5384615384615383]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_maximize_2v_4c_2o(self):
        m = Model(print_obj={})

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
        real_sol = [1.5, 1, 9]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_maximize_4v_3c_3o(self):
        m = Model(print_obj={})

        x = []
        for i in range(1,5):
            x.append(m.add_var("real+", name="x_%d" % i))


        m.maximize(3*x[1]+x[2]+4*x[3])

        m.add_constraint(x[0]+x[1]+x[2]+x[3] <= 40)
        m.add_constraint(2*x[0]+x[1]+(-1*x[2])+(-1*x[3]) <= 10)
        m.add_constraint(x[3]+(-1*x[1]) <= 10)

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [0, 15.0, 0, 25.0, 145.0]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_minimize_2v_3c_2o(self):
        m = Model(print_obj={})

        x = []
        for i in range(1,3):
            x.append(m.add_var("real+", name="x_%d" % i))

        m.minimize(0.12*x[0]+0.15*x[1])

        m.add_constraint(60*x[0]+60*x[1] >= 300)
        m.add_constraint(12*x[0]+ 6*x[1] >= 36)
        m.add_constraint(10*x[0]+30*x[1] >= 90)

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [3, 2, 0.66]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx],real_sol[x_idx])

    def test_unbound(self):
        m = Model(print_obj={})

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

    def test_diet(self):
        m = Model()

        a = m.add_var("real+", name="oat")
        b = m.add_var("real+", name="chicken")
        c = m.add_var("real+", name="egg")
        d = m.add_var("real+", name="milk")
        e = m.add_var("real+", name="cake")
        f = m.add_var("real+", name="bean")

        m.minimize(25 * a + 130 * b + 85 * c + 70 * d + 95 * e + 98 * f)

        # calories
        m.add_constraint(110 * a + 205 * b + 160 * c + 160 * d + 420 * e + 260 * f >= 2000)
        # proteins
        m.add_constraint(4 * a + 32 * b + 13 * c + 8 * d + 4 * e + 14 * f >= 55)
        # calcium
        m.add_constraint(2 * a + 12 * b + 54 * c + 285 * d + 22 * e + 80 * f >= 800)

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

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [4.0, 0, 0, 3.875, 1, 2, 662.25]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_diet_integer(self):
        m = Model(dtype="fraction")

        a = m.add_var("int+", name="oat")
        b = m.add_var("int+", name="chicken")
        c = m.add_var("int+", name="egg")
        d = m.add_var("int+", name="milk")
        e = m.add_var("int+", name="cake")
        f = m.add_var("int+", name="bean")

        m.minimize(25 * a + 130 * b + 85 * c + 70 * d + 95 * e + 98 * f)

        # calories
        m.add_constraint(110 * a + 205 * b + 160 * c + 160 * d + 420 * e + 260 * f >= 2000)
        # proteins
        m.add_constraint(4 * a + 32 * b + 13 * c + 8 * d + 4 * e + 14 * f >= 55)
        # calcium
        m.add_constraint(2 * a + 12 * b + 54 * c + 285 * d + 22 * e + 80 * f >= 800)


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

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [4, 0, 0, 4, 1, 2, 671]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_diet_integer_more_restrictions(self):
        m = Model(dtype="fraction")

        a = m.add_var("int+", name="oat")
        b = m.add_var("int+", name="chicken")
        c = m.add_var("int+", name="egg")
        d = m.add_var("int+", name="milk")
        e = m.add_var("int+", name="cake")
        f = m.add_var("int+", name="bean")

        m.minimize(25 * a + 130 * b + 85 * c + 70 * d + 95 * e + 98 * f)

        # calories
        m.add_constraint(110 * a + 205 * b + 160 * c + 160 * d + 420 * e + 260 * f >= 2000)
        # proteins
        m.add_constraint(4 * a + 32 * b + 13 * c + 8 * d + 4 * e + 14 * f >= 55)
        # calcium
        m.add_constraint(2 * a + 12 * b + 54 * c + 285 * d + 22 * e + 80 * f >= 800)


        # oats
        m.add_constraint(a <= 2)
        # chicken
        m.add_constraint(b <= 3)
        # egg
        m.add_constraint(c <= 2)
        # milk
        m.add_constraint(d <= 2)
        # cake
        m.add_constraint(e <= 1)
        # bean
        m.add_constraint(f <= 2)

        m.solve()

        computed_solution = m.get_solution_object()
        real_sol = [2, 1, 2, 2, 1, 2, 781]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_diet_10n_10i(self):
        from Model.model import Model

        m = Model()


        MIN_REQ = load_obj('diet_10n_min_req')
        ingredients, list_of_ingredients = load_obj('diet_10n_10i_ing'), load_obj('diet_10n_10i_l_o_ing')

        x = []
        for ing in list_of_ingredients:
            x.append(m.add_var("real+", name=ing))
        x = np.array(x)

        m.minimize(sum(get_by_key(ingredients, "price", list_of_ingredients) * x))

        for cst in MIN_REQ:
            left = get_by_key(ingredients, cst, list_of_ingredients)
            m.add_constraint(sum(left * x) >= MIN_REQ[cst])


        m.solve(consider_dual=0)

        try:
            i = 0
            for ing in list_of_ingredients:
                m.add_lazy_constraint(x[i] <= ingredients[ing]['max'])
                i += 1
        except InfeasibleError as e:
            pass
        else:
            self.fail("Should raise InfeasibleError but didn't")

    def test_woody_max_add_variable(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")

        m.maximize(35 * x1 + 60 * x2)

        m.add_constraint(8 * x1 + 12 * x2 <= 120)
        m.add_constraint(15 * x2 <= 60)
        m.add_constraint(3*x1+6*x2 <= 48)

        m.solve(consider_dual=0)
        m.add_new_variable([16,20,9],75)

        computed_solution = m.get_solution_object()
        real_sol = [12, 2, 0, 540]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_woody_min_add_variable(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")

        m.minimize(35 * x1 + 60 * x2)

        m.add_constraint(8 * x1 + 12 * x2 >= 120)
        m.add_constraint(15 * x2 >= 60)
        m.add_constraint(3*x1+6*x2 >= 48)

        m.solve(consider_dual=0)
        m.add_new_variable([16,20,9],75)

        computed_solution = m.get_solution_object()
        print(m.get_solution_object())
        real_sol = [9, 0, 3,540]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_woody_max_lazy_constraint(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")
        x3 = m.add_var("real+", name="c")

        m.maximize(35 * x1 + 60 * x2 + 75 * x3)

        m.add_constraint(8 * x1 + 12 * x2 + 16 * x3 <= 120)
        m.add_constraint(15 * x2 + 20 * x3 <= 60)
        m.add_constraint(3*x1+6*x2+9*x3 <= 48)

        m.solve(consider_dual=0)
        m.add_lazy_constraint(x1 <= 5)
        m.add_lazy_constraint(x3 >= 1)

        computed_solution = m.get_solution_object()
        real_sol = [5.0, 2.6666666666666665, 1.0, 410]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_woody_min_lazy_constraint(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")
        x3 = m.add_var("real+", name="c")

        m.minimize(35 * x1 + 60 * x2 + 75 * x3)

        m.add_constraint(8 * x1 + 12 * x2 + 16 * x3 >= 120)
        m.add_constraint(15 * x2 + 20 * x3 >= 60)
        m.add_constraint(3*x1+6*x2+9*x3 >= 48)

        m.solve(consider_dual=0)
        m.add_lazy_constraint(x1 <= 5)
        m.add_lazy_constraint(x3 >= 1)

        computed_solution = m.get_solution_object()
        real_sol = [5.0, 0, 5.0, 550]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_woody_min_dual_add_variable(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")

        m.minimize(35 * x1 + 60 * x2)

        m.add_constraint(8 * x1 + 12 * x2 >= 120)
        m.add_constraint(15 * x2 >= 60)
        m.add_constraint(3 * x1 + 6 * x2 >= 48)

        m.solve(consider_dual=2)
        m.add_new_variable([16, 20, 9], 75)

        computed_solution = m.get_solution_object()
        print(m.get_solution_object())
        real_sol = [9, 0, 3, 540]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_woody_max_dual_add_variable(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")

        m.maximize(35 * x1 + 60 * x2)

        m.add_constraint(8 * x1 + 12 * x2 <= 120)
        m.add_constraint(15 * x2 <= 60)
        m.add_constraint(3 * x1 + 6 * x2 <= 48)

        m.solve(consider_dual=2)
        m.add_new_variable([16, 20, 9], 75)

        computed_solution = m.get_solution_object()
        real_sol = [12, 2, 0, 540]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_woody_max_dual_lazy_constraint(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")
        x3 = m.add_var("real+", name="c")

        m.maximize(35 * x1 + 60 * x2 + 75 * x3)

        m.add_constraint(8 * x1 + 12 * x2 + 16 * x3 <= 120)
        m.add_constraint(15 * x2 + 20 * x3 <= 60)
        m.add_constraint(3 * x1 + 6 * x2 + 9 * x3 <= 48)

        m.solve(consider_dual=2)
        m.add_lazy_constraint(x1 <= 5)
        m.add_lazy_constraint(x3 >= 1)

        computed_solution = m.get_solution_object()
        real_sol = [5, 2.6666666666666665, 1, 410]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_woody_min_dual_lazy_constraint(self):
        m = Model()

        x1 = m.add_var("real+", name="a")
        x2 = m.add_var("real+", name="b")
        x3 = m.add_var("real+", name="c")

        m.minimize(35 * x1 + 60 * x2 + 75 * x3)

        m.add_constraint(8 * x1 + 12 * x2 + 16 * x3 >= 120)
        m.add_constraint(15 * x2 + 20 * x3 >= 60)
        m.add_constraint(3 * x1 + 6 * x2 + 9 * x3 >= 48)

        m.solve(consider_dual=2)
        m.add_lazy_constraint(x1 <= 5)
        m.add_lazy_constraint(x3 >= 1)

        computed_solution = m.get_solution_object()
        real_sol = [5, 0, 5, 550]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

if __name__ == '__main__':
    unittest.main()
    # ti = time()
    # t = MyTest()
    # t.test_diet_integer_more_restrictions()
    # print("Time", time()-ti)
    # t.test_woody_min_dual_add_variable()
