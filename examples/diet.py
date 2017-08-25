import numpy as np
from time import time
import random, string

from Model.model import Model

m = Model(print_obj={
   'start_conf': True,
   'end_conf': True
})

def get_keys(obj):
    keys = []
    for ing in obj:
        keys.append(ing)
    return keys

def get_by_key(obj,key, key_list=False):
    arr = []
    if key_list:
        for ing in key_list:
            arr.append(obj[ing][key])
    else:
        for ing in obj:
            arr.append(obj[ing][key])
    return np.array(arr)

ingredients = {
    "oat": {
        "kcal": 110,
        "protein": 4,
        "calcium": 2,
        "price": 25,
        "max": 4
    },
    "chicken": {
        "kcal": 205,
        "protein": 32,
        "calcium": 12,
        "price": 130,
        "max": 3
    },
    "egg": {
        "kcal": 160,
        "protein": 13,
        "calcium": 54,
        "price": 85,
        "max": 2
    },
    "milk": {
        "kcal": 160,
        "protein": 8,
        "calcium": 285,
        "price": 70,
        "max": 8
    },
    "cake": {
        "kcal": 420,
        "protein": 4,
        "calcium": 22,
        "price": 95,
        "max": 1
    },
    "bean": {
        "kcal": 260,
        "protein": 14,
        "calcium": 80,
        "price": 98,
        "max": 2
    }
}

MIN_REQ = {
    "kcal": 2000,
    "protein": 55,
    "calcium": 800,
}

list_of_ingredients = get_keys(ingredients)

x = []
for ing in list_of_ingredients:
    x.append(m.add_var("real+", name=ing, ub=ingredients[ing]["max"]))
x = np.array(x)

m.minimize(sum(get_by_key(ingredients,"price", list_of_ingredients)*x))

for cst in MIN_REQ:
    left = get_by_key(ingredients,cst, list_of_ingredients)
    m.add_constraint(sum(left*x) >= MIN_REQ[cst])


print("all added")

t0 = time()

m.solve(consider_dual=0)
print("Solved first in %f" % (time()-t0))

m.print_solution(slack=False)



sol_obj = m.get_solution_object()
solved = False
while not solved:
    solved = True
    i = 0
    for ing in list_of_ingredients:
        if sol_obj[i] > ingredients[ing]['max']:
            solved = False
            m.add_lazy_constraint(x[i] <= ingredients[ing]['max'])
            exit(1)
            sol_obj = m.get_solution_object()
            break
        i += 1

print("Solved total in %f" % (time()-t0))

print("End tableau")
print(m.t.tableau)
print("Steps: ", m.steps)

m.print_solution(slack=False)


