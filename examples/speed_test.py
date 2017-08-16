import numpy as np
from time import time
import random, string
random.seed(9001)

from Model.model import Model

m = Model(print_obj={
    'information': True,
    'timing': True
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
        "vitB12": 10,
        "vitC": 12,
        "price": 25,
        "max": 4
    },
    "chicken": {
        "kcal": 205,
        "protein": 32,
        "calcium": 12,
        "vitB12": 5,
        "vitC": 2,
        "price": 130,
        "max": 3
    },
    "egg": {
        "kcal": 160,
        "protein": 13,
        "calcium": 54,
        "vitB12": 5,
        "vitC": 2,
        "price": 85,
        "max": 2
    },
    "milk": {
        "kcal": 160,
        "protein": 8,
        "calcium": 285,
        "vitB12": 5,
        "vitC": 2,
        "price": 70,
        "max": 8
    },
    "cake": {
        "kcal": 420,
        "protein": 4,
        "calcium": 22,
        "vitB12": 5,
        "vitC": 2,
        "price": 95,
        "max": 1
    },
    "bean": {
        "kcal": 260,
        "protein": 14,
        "calcium": 80,
        "vitB12": 5,
        "vitC": 2,
        "price": 98,
        "max": 2
    }
}

MIN_REQ = {
    "kcal": 2000,
    "protein": 55,
    "calcium": 800,
    "vitB12": 100,
    "vitC": 130,
}

MIN_REQ = {}

ingredients = {}
for j in range(1200):
    nut = "".join([random.choice(string.ascii_letters) for d in range(10)])
    MIN_REQ[nut] = random.randint(50,2000)

for i in range(100):
    ing = "".join( [random.choice(string.ascii_letters) for d in range(15)] )
    ingredients[ing] = {}
    for nut in MIN_REQ:
        nut_val = random.randint(MIN_REQ[nut]//100,MIN_REQ[nut]//10)
        ingredients[ing][nut] = nut_val
    ingredients[ing]["price"] = random.randint(15,90)
    ingredients[ing]["max"] = random.randint(1,3)

list_of_ingredients = get_keys(ingredients)

x = []
for ing in list_of_ingredients:
    x.append(m.add_var("real+", name=ing, ub=ingredients[ing]["max"]))
x = np.array(x)

m.minimize(sum(get_by_key(ingredients,"price", list_of_ingredients)*x))

for cst in MIN_REQ:
    left = get_by_key(ingredients,cst, list_of_ingredients)
    m.add_constraint(sum(left*x) >= MIN_REQ[cst])

"""
i = 0
for ing in list_of_ingredients:
    m.add_constraint(x[i] <= ingredients[ing]['max'])
    i += 1
"""
print("all added")

t0 = time()
m.solve()
print("Solved in %f" % (time()-t0))

m.print_solution(slack=False)
print("Steps: ",m.steps)
