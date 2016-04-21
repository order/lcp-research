import numpy as np

def random_points(limits,N):
    D = len(limits)
    points = np.empty((N,D))
    for (i,(l,u)) in enumerate(limits):
        points[:,i] = np.random.uniform(l,u,N)
    return points
