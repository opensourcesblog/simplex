import unittest
from inspect import ismethod
from Model.model import Model
import numpy as np
from Model.error import *
import random, string


def get_keys(obj):
    keys = []
    for ing in obj:
        keys.append(ing)
    return keys


def get_by_key(obj, key, key_list=False):
    arr = []
    if key_list:
        for ing in key_list:
            arr.append(obj[ing][key])
    else:
        for ing in obj:
            arr.append(obj[ing][key])
    return np.array(arr)

def fill_MIN_REQ(n):
    MIN_REQ = {}
    for j in range(n):
        nut = "".join([random.choice(string.ascii_letters) for d in range(10)])
        MIN_REQ[nut] = random.randint(50, 2000)
    return MIN_REQ

def fill_ING(MIN_REQ,n,vals=False):
    if vals is False:
        vals = {'max': 4}
    ingredients = {}
    list_of_ingredients = []
    for i in range(n):
        ing = "".join([random.choice(string.ascii_letters) for d in range(15)])
        ingredients[ing] = {}
        for nut in MIN_REQ:
            nut_val = random.randint(MIN_REQ[nut] // 100, MIN_REQ[nut] // 10)
            ingredients[ing][nut] = nut_val
        ingredients[ing]["price"] = random.randint(15, 90)
        ingredients[ing]["max"] = random.randint(1, vals['max'])
        list_of_ingredients.append(ing)
    return ingredients, list_of_ingredients

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
        real_sol = [4.0, 0, 0, 3.8750000000005276, 1, 2, 662.25000000001796]
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_diet_100n_2000i(self):
        random.seed(9001)

        from Model.model import Model

        m = Model(print_obj={
            'timing': True
        })

        MIN_REQ = fill_MIN_REQ(100)
        ingredients, list_of_ingredients = fill_ING(MIN_REQ,2000, {'max': 4})


        x = []
        for ing in list_of_ingredients:
            x.append(m.add_var("real+", name=ing, ub=ingredients[ing]["max"]))
        x = np.array(x)

        m.minimize(sum(get_by_key(ingredients, "price", list_of_ingredients) * x))

        for cst in MIN_REQ:
            left = get_by_key(ingredients, cst, list_of_ingredients)
            m.add_constraint(sum(left * x) >= MIN_REQ[cst])


        m.solve(consider_dual=0)

        sol_obj = m.get_solution_object()

        solved = False
        while not solved:
            solved = True
            i = 0
            for ing in list_of_ingredients:
                if sol_obj[i] > ingredients[ing]['max']:
                    solved = False
                    m.add_lazy_constraint(x[i] <= ingredients[ing]['max'])
                    sol_obj = m.get_solution_object()
                    break
                i += 1

        computed_solution = m.get_solution_object()
        real_sol = np.load('test_obj/diet_100n_2000i.npy')
        for x_idx in range(len(real_sol)):
            self.assertAlmostEqual(computed_solution[x_idx], real_sol[x_idx])

    def test_diet_10n_10i(self):
        random.seed(9001)

        from Model.model import Model

        m = Model(print_obj={
            'timing': True
        })


        MIN_REQ = fill_MIN_REQ(10)
        ingredients, list_of_ingredients = fill_ING(MIN_REQ,10, {'max': 3})


        x = []
        for ing in list_of_ingredients:
            x.append(m.add_var("real+", name=ing, ub=ingredients[ing]["max"]))
        x = np.array(x)

        m.minimize(sum(get_by_key(ingredients, "price", list_of_ingredients) * x))

        for cst in MIN_REQ:
            left = get_by_key(ingredients, cst, list_of_ingredients)
            m.add_constraint(sum(left * x) >= MIN_REQ[cst])


        m.solve(consider_dual=0)

        sol_obj = m.get_solution_object()

        try:
            solved = False
            while not solved:
                solved = True
                i = 0
                for ing in list_of_ingredients:
                    if sol_obj[i] > ingredients[ing]['max']:
                        solved = False
                        m.add_lazy_constraint(x[i] <= ingredients[ing]['max'])
                        sol_obj = m.get_solution_object()
                        break
                    i += 1
        except InfeasibleError as e:
            pass
        else:
            self.fail("Should raise InfeasibleError but didn't")

if __name__ == '__main__':
    unittest.main()
    # t = MyTest()
    # t.test_minimize_2v_3c_2o()
