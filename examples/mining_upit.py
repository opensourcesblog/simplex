import numpy as np
from time import time
import os

from Model.model import Model

m = Model(print_obj={
   # 'start_conf': True,
   # 'end_conf': True
    "pivot": True,
    # "timing": True,
    "save_tab": True
})

instance = 'examples/data/newman1'

def read_cpit(dataset):
    f = open(dataset + ".cpit", 'r')
    current_parameter = "NULL"
    profit = []
    res_constr_coeff = []
    n_blocks = 0
    n_periods = 0
    res_constr_limits = []

    for line in f:
        split_line = [x for x in line.split()]

        if current_parameter == "RESOURCE_CONSTRAINT_COEFFICIENTS":
            if (split_line[0] != "EOF"):
                res_constr_coeff[int(split_line[0])][int(split_line[1])] = float(split_line[2])
            else:
                break

        if current_parameter == "OBJECTIVE_FUNCTION":
            if split_line[0] == "RESOURCE_CONSTRAINT_COEFFICIENTS:":
                current_parameter = "RESOURCE_CONSTRAINT_COEFFICIENTS"
                res_constr_coeff = [[0 for j in range(n_res_constr)] for i in range(n_blocks)]
            else:
                profit.append(float(split_line[1]))

        if current_parameter == "RESOURCE_CONSTRAINT_LIMITS":
            if split_line[0] == "OBJECTIVE_FUNCTION:":
                current_parameter = "OBJECTIVE_FUNCTION"
            else:
                res_constr_limits[int(split_line[0])][int(split_line[1])] = float(split_line[3])

        if current_parameter == "NULL":
            if split_line[0] == "NBLOCKS:":
                n_blocks = int(split_line[1])
            if split_line[0] == "NPERIODS:":
                n_periods = int(split_line[1])
            if split_line[0] == "NRESOURCE_SIDE_CONSTRAINTS:":
                n_res_constr = int(split_line[1])
            if split_line[0] == "DISCOUNT_RATE:":
                discount_rate = float(split_line[1])
            if split_line[0] == "RESOURCE_CONSTRAINT_LIMITS:":
                current_parameter = "RESOURCE_CONSTRAINT_LIMITS"
                res_constr_limits = [[0 for a in range(n_periods)] for b in range(n_res_constr)]

    f.close()
    return n_blocks, n_periods, n_res_constr, discount_rate, res_constr_limits, profit, res_constr_coeff


def read_blocks(dataset):
    f = open(dataset + ".blocks", 'r')
    x_value, y_value, z_value = [],[],[]
    for line in f:
        split_line = [x for x in line.split()]
        x_value.append(int(split_line[1]))
        y_value.append(int(split_line[2]))
        z_value.append(int(split_line[3]))
    f.close()
    return x_value, y_value, z_value


def read_prec(dataset, n_blocks):
    f = open(dataset + ".prec", 'r')
    pred = [[] for i in range(n_blocks)]
    for line in f:
        split_line = [int(x) for x in line.split()]
        block_id = split_line[0]
        n_pred = split_line[1]
        for i in range(n_pred):
            pred[block_id].append(split_line[i + 2])
    f.close()
    return pred

def get_by_key(arr,key):
    result = []
    for i in arr:
        result.append(i[key])
    return np.array(result)


n_blocks, n_periods, n_res_constr, discount_rate, res_constr_limits, profit, res_constr_coeff = read_cpit(instance)
x_value, y_value, z_value = read_blocks(instance)
pred = read_prec(instance, n_blocks)

print("#blocks", n_blocks)
print("#periods", n_periods)

x = []
for i in range(n_blocks):
    x.append(m.add_var("int+", name=i))
x = np.array(x)

m.file_name = "examples/data/newman"
m.maximize(sum(profit*x))

# binary
for i in range(n_blocks):
    m.add_constraint(x[i] <= 1)

# cost
# m.add_constraint(sum(get_by_key(blocks,"c")*x) <= max_c)

for i in range(2,n_blocks):
    if len(pred[i]) > 0:
        m.add_constraint(len(pred[i])*x[i]-sum(x[pred[i]]) <= 0)

print("all added")

t0 = time()

m.solve(tableau_file=m.file_name)
print("Solved first in %f" % (time()-t0))

m.print_solution(slack=False)



