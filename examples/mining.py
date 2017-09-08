import numpy as np
from time import time
import random, string

from Model.model import Model

m = Model(print_obj={
   'start_conf': True,
   'end_conf': True
})


def get_by_key(arr,key):
    result = []
    for i in arr:
        result.append(i[key])
    return np.array(result)

blocks = [
            {"pred": [], 'p': -100, 'c': 100},
            {"pred": [], 'p': -150, 'c': 200},
            {"pred": [0,1], 'p': -100, 'c': 100},
            {"pred": [0,1], 'p': 250, 'c': 300},
            {"pred": [1,2], 'p': 300, 'c': 100},
            {"pred": [2,3], 'p': 1000, 'c': 1000},
            {"pred": [4,5], 'p': 10000, 'c': 300},
            {"pred": [4,5,6], 'p': 15000, 'c': 1000},
        ]

max_c = 2000


x = []
for i in range(len(blocks)):
    x.append(m.add_var("int+", name=i))
x = np.array(x)

m.maximize(sum(get_by_key(blocks,"p")*x))

# binary
for i in range(len(blocks)):
    m.add_constraint(x[i] <= 1)

# cost
m.add_constraint(sum(get_by_key(blocks,"c")*x) <= max_c)


for i in range(len(blocks)):
    if len(blocks[i]["pred"]) > 0:
        m.add_constraint(len(blocks[i]["pred"])*x[i]-sum(x[blocks[i]["pred"]]) <= 0)

print("all added")

t0 = time()

m.solve()
print("Solved first in %f" % (time()-t0))

m.print_solution(slack=False)



