[![Build Status](https://travis-ci.org/opensourcesblog/simplex.svg?branch=master)](https://travis-ci.org/opensourcesblog/simplex)
# Simplex

How the simplex algorithm works is described in my blog article on [opensourc.es](http://opensourc.es/blog/simplex).

There are several examples to show you how to use the algorithm in the `examples` folder.

You can start the diet example using 
```
python -m examples.diet
```

Python3 is recommended for this project. All tests are run on Python >= 3.4. 


The general structure of each example looks like this:

```
m = Model()
```
Adding variables to the model:
```
a = m.add_var("real+", name="apple")
b = m.add_var("real+", name="banana")
c = m.add_var("real+", name="carrot")
```
Include an objective function:
```
m.maximize(45*a+60*b+50*c)
```
And several constraints:
```
m.add_constraint(20*a+10*b+10*c <= 2400)
m.add_constraint(12*a+28*b+16*c <= 2400)
m.add_constraint(15*a+6*b+16*c <= 2400)
m.add_constraint(10*a+15*b <= 2400)
m.add_constraint(a <= 100)
m.add_constraint(b <= 40)
m.add_constraint(c <= 60)
```

At the end you have to use the method `solve` and print the solution:
```
m.solve()
m.print_solution()
```
