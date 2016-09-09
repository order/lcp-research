import numpy as np
import itertools

fh = open('test.geo','w')

pid = 0
for coord in itertools.product([-1,1],repeat=3):
    fh.write('Point({0})={{{1},{2},{3}}}\n'.format(pid,
                                                   coord[0],
                                                   coord[1],
                                                   coord[2]))
    pid += 1

lid = 
fh.close()
