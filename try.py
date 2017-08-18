import numpy as np

np.set_printoptions(precision=2,
                       threshold=100000,
                       linewidth=600,
                       suppress=True)

a = [
    [0, 0, 0, -10, 15/4, 1, -10, 75],
    [0, 0, 1, 2, -1/4,0, 2/3,  -1],
    [0, 1, 0, -1, 1/2, 0, -1,  18],
    [1, 0, 0, 10,5/2, 0, 5, 570]
]
a = np.array(a)

print(a)
a[1] *= -4
print(a)
for i in [0,2,3]:
    fac = -a[i][4]/a[1][4]
    a[i] += fac*a[1]
print(a)
