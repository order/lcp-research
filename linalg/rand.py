import numpy as np

def rademacher(**kwargs):
    shape = kwargs.get('shape',1)
    return 2*np.random.randint(2,size=shape) - 1

def random_points(limits,N):
    D = len(limits)
    points = np.empty((N,D))
    for (i,(l,u)) in enumerate(limits):
        points[:,i] = np.random.uniform(l,u,N)
    return points
