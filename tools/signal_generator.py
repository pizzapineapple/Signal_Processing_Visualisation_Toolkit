from numpy import linspace
import numpy
from math import pi

import math
from matplotlib import pyplot

x = linspace(0,100, 100000)
a = 10*numpy.sin(x)
b = 10*numpy.sin(x+1/128)
c = a-b

for i, j, k in zip(a, b, c):
    print(k, ", ", i)