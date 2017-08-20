import numpy as np


class Tableau(object):
    def __init__(self):
        self._data = np.zeros((1,1))

    def __set__(self, obj, val):
        print("set")
        self._data = val

    def __get__(self, obj, obj_type):
        print("get")
        return self._data

    def get_obj(self,tv):
        return self._data[-1]

    def set_obj(self, tv, value):
        self._data[-1] = value


class TableauView(object):
    tableau = Tableau()
    obj = property(tableau.get_obj,tableau.set_obj)

t = TableauView()
t.tableau = np.zeros((5,5))
t.tableau[-1] = t.tableau[-1]+2
t.tableau = np.c_[t.tableau[:,:2], [0,0,0,1,1], t.tableau[:,2:]]
t.obj = 3
print(t.tableau)
print(t.obj)
