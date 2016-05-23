import numpy as np
import cdiscrete as cd
from discrete import make_points
import matplotlib.pyplot as plt

low = np.zeros(2)
high = np.array([2.0,1.0],dtype='double')
num = np.array([2,1],dtype='uint64')

v = np.zeros((3,2))
v[1,1] = 1
v = np.hstack([v.flatten(),-1])
print v

P = make_points([np.linspace(low[d],high[d],num[d]+1) for d in xrange(2)])
print P

I = cd.interpolate(v,P,low,high,num)
print I
