import numpy as np
import bisect
import math
import scipy.sparse
import matplotlib.pyplot as plt

# Transforms a coordinate into flanking indices
# ASSUMES p_grid is a regular grid
def regular_index_transform(p,p_grid):
    """
    Takes in a 1D coordinate p, and a sorted list of cut points p_grid
    Returns an ordered pair (GS,LL) where GS is the greatest element in p_grid smaller than p,
    And LL is the least element larger than p.
    If p is exactly a cut-point, just return that
    """
    if p < p_grid[0] or p > p_grid[-1]:
        return None

    span = p_grid[-1] - p_grid[0]
    min_el = p_grid[0]
    
    i = math.floor((p - min_el) * float(len(p_grid) - 1) / span)
    if i + 1 < p_grid.size:
        return [i,i+1]
    return [i]

def binary2unaryindex(x,y,N):
    return x + y*N
def unary2binaryindex(i,N):
    return (i%N,i/N)

class Regular2DGrid(object):
    def __init__(self,x_grid,y_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.n_points = x_grid.size * y_grid.size
        self.points = None
        
    def index2point(self,i):
        x_grid = self.x_grid
        y_grid = self.y_grid
        Nx = x_grid.size

        (x_index,y_index) = unary2binaryindex(i, Nx)

        return (x_grid[x_index],y_grid[y_index])

    def point_gen(self):
        for i in xrange(self.n_points):
            yield self.index2points(i)

    # Generating function for the points
    def get_points(self):
        if not self.points is None:
            return self.points
        X,Y = np.meshgrid(self.x_grid,self.y_grid)
        self.points = np.column_stack((X.flatten(),Y.flatten()))
        return self.points

    # Transforms a 2D point into the 4 weights for neighbouring indices
    # ASSUMES a regular grid structure
    def point2bilinear(self,p):
        x_grid = self.x_grid
        y_grid = self.y_grid
        Nx = x_grid.size

        xs = regular_index_transform(p[0],x_grid)
        ys = regular_index_transform(p[1],y_grid)
        if None == xs or None == ys:
            return None

        assert(x_grid[xs[0]] <= p[0])
        assert(y_grid[ys[0]] <= p[1])
        assert(len(xs) < 2 or x_grid[xs[1]] >= p[0])
        assert(len(ys) < 2 or y_grid[ys[1]] >= p[1])


        weights = {}
        for i in xrange(len(xs)):
            if len(xs) == 1:
                wx = 1.0
            else:
                h = x_grid[xs[1]] - x_grid[xs[0]]
                wx = abs(p[0] - x_grid[xs[1-i]]) / h
            for j in xrange(len(ys)):
                if len(ys) == 1:
                    wy = 1.0
                else:
                    h = y_grid[ys[1]] - y_grid[ys[0]]
                    wy = abs(p[1] - y_grid[ys[1-j]]) / h

                index = binary2unaryindex(xs[i],ys[j],Nx)
                weights[index] = wx*wy
        return weights

    def points2matrix(self,points):
        [N,d] = points.shape
        
        assert(d == 2)
        
        # Use DOK for maximum flexibility
        A = scipy.sparse.dok_matrix((self.n_points,N))
        OOB = []
        for i in xrange(N):
            weights = self.point2bilinear(points[i,:])
            if None == weights:
                OOB.append(i)
                continue
            for (j,w) in weights.items():
                A[j,i] = w
        return (A,OOB)        
