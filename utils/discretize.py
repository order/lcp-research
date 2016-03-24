import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class RegularGridPartition(object):
    """
    Defines an grid split based on irregular cutpoints
    """
    def __init__(self,grid_desc):
        D = len(grid_desc)
        self.grid_desc = grid_desc # List of (low,high,num) triples

        self.dim = D
        self.num_grid = np.array([n for (l,h,n) in self.grid_desc])
        self.num_cells = np.prod(self.num_grid)
        self.skip = np.array([(h-l) / n for (l,h,n) in self.grid_desc])
                
    def dim_cell_boundary(self,d,k):
        (l,h,n) = self.grid_desc[d]
        assert(0 <= k < n)
        lo = l + k*self.skip[d]
        hi = lo + self.skip[d]
        return (lo,hi)
    
    def cell_boundary(self,coord):
        """
        Get the cell boundary of the cell at given coordinates
        Can specify None in some coordinates to avoid lookup
        """
        assert(self.dim == len(coord))
        cell = []
        for d in xrange(self.dim):
            if not coord[d]:
                cell.append(None)
                continue
            bound = self.dim_cell_boundary(d,coord[d])
            cell.append(bound)
        return cell
    
class IrregularGridPartition(object):
    """
    Defines an grid split based on irregular cutpoints
    """
    def __init__(self,cpts):
        """
        Cutpoints is a D long list of cut points for
        each dimension.
        So [[0,1],[0,0.6,1]] describes two cells:
        [0,1] x [0,0.6] and [0,1] x [0.6,1]
        """
        D = len(cpts)
        self.dim = D
        self.cpts = [np.array(c) for c in cpts]
        self.check_cutpoints()
        
        self.num_cpts = [len(x) for x in cpts]
        self.num_cells = np.prod([len(x)-1 for x in cpts])
        self.boundary = [(cpts[d][0],cpts[d][-1]) for d in xrange(D)]

    def check_cutpoints(self):
        for d in xrange(self.dim):
            K = len(self.cpts[d])
            assert(K > 1) # At least one interval
            for i in xrange(K-1):
                # Monotonically increasing
                assert(self.cpts[d][i] < self.cpts[d][i+1])
                
    def dim_cell_boundary(self,d,k):
        assert(0 <= k < self.num_cpts[d]-1)
        lo = self.cpts[d][k]
        hi = self.cpts[d][k+1]
        return (lo,hi)
    
    def cell_boundary(self,coord):
        """
        Get the cell boundary of the cell at given coordinates
        Can specify None in some coordinates to avoid lookup
        """
        assert(self.dim == len(coord))
        cell = []
        for d in xrange(self.dim):
            if not coord[d]:
                cell.append(None)
                continue
            bound = self.dim_cell_boundary(d,coord[d])
            cell.append(bound)
        return cell

    def boundary(self):
        return self.boundary

####################################
# REGULAR GRID DISCRETIZER

class RegularGridDiscretizer(object):
    def __init__(self,grid_partition):
        self.partition = grid_partition
        self.num_cells = grid_partition.num_cells
        self.fuzz = 1e-12

    def to_cell_coord(self,points):
        (N,D) = points.shape
        part = self.partition

        assert(D == part.dim)
        Coords = np.empty((N,D))
        for d in xrange(D):
            (low,high,n) = part.grid_desc[d]
            # Linearly transform the data so [low,high) will be in [0,n)
            transform = n * (points[:,d] - low) / (high - low)
            Coords[:,d] = np.floor(transform)

            # Fuzz top boundary to get [low,high]
            fuzz_mask = np.logical_and(high <= points[:,d],
                                     points[:,d] < high + self.fuzz)
            Coords[fuzz_mask,d] = n-1

            oob_mask = np.logical_or(low > points[:,d],
                                     points[:,d] > high + self.fuzz)
            Coords[oob_mask,d] = np.nan            
        return Coords
    
####################################
# IRREGULAR GRID DISCRETIZER

class IrregularGridDiscretizer(object):
    def __init__(self,grid_partition):
        self.partition = grid_partition
        self.num_cells = grid_partition.num_cells
        self.fuzz = 1e-12
        
    def to_cell_coord(self,points):
        part = self.partition # slang, yo
        (N,D) = points.shape
        assert(D == part.dim)

        Coords = np.empty((N,D))
        for d in xrange(D):
            K = part.num_cpts[d]
            coord = np.searchsorted(part.cpts[d],
                                    points[:,d],
                                    side='right') - 1
            # The 'right' is important if points are exactly
            # cut points
            Coords[:,d] = coord
                
            hi = part.cpts[d][-1]
            fuzz_mask = np.where(np.logical_and(points[:,d] >= hi,
                                              points[:,d] < hi+self.fuzz))
            Coords[fuzz_mask,d] = K-2
            

            (left,right) = part.boundary[d]
            oob_mask = np.logical_or(points[:,d] < left,
                                     points[:,d] >= right + self.fuzz)
            Coords[oob_mask,d] = np.nan
        return Coords
    
    def lower_corner(self,Coords):
        (N,D) = Coords.shape
        Corner = np.empty(Coords.shape)
        Cuts = self.partition.cpts
        for d in xrange(D):
            Corner[:,d] = Cuts[d][Coords[:,d].astype('int')]
        return Corner
    
    def upper_corner(self,Coords):
        return self.lower_corner(Coords+1)
    
    def midpoints(self,Coords):
        return 0.5*self.lower_corner(Coords)\
            +0.5*self.upper_corner(Coords)
    
class Indexer(object):
    def __init__(self,lens,order='C'):
        lens = np.array(lens)
        (D,) = lens.shape
        self.dim = D
        
        self.max_index = np.product(lens)
        self.lens = np.array(lens)
        
        """
        Assume C ordering most of the time.
        Fortran style just for completeness
        """
        assert(order=='C' or order=='F')
        self.order = order
        if order == 'F':
            # Fortran order; first coord changes index least
            coef = np.cumprod(lens)
            coef = np.roll(coef,1)
            coef[0] = 1.0
        else:
            # C order; last coord changes index least
            # Need to flip the coefficients around to
            # make the C order            
            coef = np.cumprod(np.flipud(lens))
            coef = np.roll(coef,1)
            coef[0] = 1.0
            coef = np.flipud(coef)            
        self.coef = coef
        
    def coords_to_indices(self,coords):
        """
        Turn D dimensional coords into indices
        """
        (N,D) = coords.shape

        assert(is_int(coords))
        assert(D == self.dim) # Dimension right
        assert(not np.any(coords < 0)) # all non
        assert(np.all(coords < self.lens))
        
        return coords.dot(self.coef)
    
    def indices_to_coords(self,indices):
        (N,) = indices.shape
        D = len(self.coef)
        
        assert(is_int(indices))     
        res = indices
        coords = np.empty((N,D))
        if 'C' == self.order:
            idx_order = xrange(D)
        else:
            idx_order = reversed(xrange(D))
        for d in idx_order:
            (coord,res) = divmod(res,self.coef[d])
            coords[:,d] = coord
        oob_mask = np.logical_or(indices < 0,
                                 indices >= self.max_index)
        coords[oob_mask] = np.nan
        return coords.astype('i')

#######################################
# AUX functions
    
def is_int(x):
    return x.dtype.kind in 'ui'

def make_points(gens,order='C'):
    """
    Makes the mesh in the order you would expect for
    np.reshape after.

    E.g. if handed [np.linspace(0,1,5),np.linspace(0,1,7)]
    then this would make the points P such that if mapped
    the 2D plot makes spacial sense. So np.reshape(np.sum(P),(5,7))
    would look pretty and not jumpy
    """
    if 'F' == order:
        gens = list(reversed(gens))
    if 1 == len(gens):
        return gens[0][:,np.newaxis] # meshgrid needs 2 args
    
    meshes = np.meshgrid(*gens,indexing='ij')
    points = np.column_stack([M.flatten() for M in meshes])
    if 'F' == order:
        return np.fliplr(points)
    return points    

#############################3
# Tests

def test_1d():    
    cuts = [[0,0.6,1]]
    grid_part = IrregularGridPartition(cuts)
    disc = IrregularGridDiscretizer(grid_part)

    N = 250
    samples = np.linspace(0,1,N)[:,np.newaxis]
    coords = disc.to_cell_coord(samples)
    assert((N,1) == coords.shape)

    for n in xrange(N):
        plt.plot([0,1],[samples[n,0],coords[n,0]],'-b',alpha=0.1)
    plt.show()

def test_index(order='C'):
    
    lens = [3,2,4,3]
    indexer = Indexer(lens,order=order)

    coords = make_points([xrange(x) for x in lens],order=order)
    indices = indexer.coords_to_indices(coords)
    backagain = indexer.indices_to_coords(indices)
    assert(np.all(coords == backagain))
    
    (N,) = indices.shape
    for n in xrange(N):
        print '{0} -> {1} -> {2}'.format(coords[n,:],
                                         indices[n],
                                         backagain[n,:])
def test_timing():
    N = 250
    M = N+1
    S = 10000
    grid_desc = [(0,1,N),(0,1,N)]
    cuts = IrregularGridPartition([np.linspace(0,1,N+1),
                                   np.linspace(0,1,N+1)])
    reg_grid = RegularGridPartition(grid_desc)
    reg_disc = RegularGridDiscretizer(reg_grid)
    irr_disc = IrregularGridDiscretizer(cuts)
    
    assert(reg_disc.num_cells == irr_disc.num_cells)

    points = make_points([np.linspace(0,1,M),np.linspace(0,1,M)])
    points = np.vstack([points,np.random.rand(S,2)])

    start = time.time()
    reg_coord = reg_disc.to_cell_coord(points)
    print 'Regular time:',time.time() - start
    start = time.time()
    irr_coord = irr_disc.to_cell_coord(points)
    print 'Irregular time:',time.time() - start

    error = np.sum(np.abs(reg_coord - irr_coord),axis=1)
    print 'Total error',np.sum(error[:]) 
    for n in xrange(M*M):
        if np.all(reg_coord[n,:] == irr_coord[n,:]):
            continue
        print '{0} -> {1},{2}'.format(points[n,:],
                                      reg_coord[n,:],
                                      irr_coord[n,:])
    assert(np.all(reg_coord == irr_coord))

def test_2d():
    N = 3
    S = 50
    cuts = IrregularGridPartition([np.linspace(0,1,N+1),
                                   np.linspace(0,1,N+1)])
    irr_disc = IrregularGridDiscretizer(cuts)
    
    points = make_points([np.linspace(0,1,S),
                          np.linspace(0,1,S)])
    coords = irr_disc.to_cell_coord(points)
    midpoints = irr_disc.midpoints(coords)
    (n,d) = midpoints.shape
    assert(d == 2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in xrange(n):
        ax.plot([points[i,0],midpoints[i,0]],
                    [points[i,1],midpoints[i,1]],
                    [0,1],'-b',alpha = 0.1)
    ax.view_init(elev=90,azim=0) # 2D view
    plt.show()
    
###############################
# Interp trial (should flesh out).
    
def trial_interpolate(X,low,high):
    (N,D) = X.shape
    assert(np.all(X >= 0))
    assert(np.all(X <= 1))
    assert((D,) == low.shape)
    assert((D,) == high.shape)
    
    Coef = np.empty((N,2*D))
    for d in xrange(D):
        Z = high[d] - low[d]
        Coef[:,2*d] = (X[:,d] - low[d])/Z
        Coef[:,2*d+1] = (high[d] - X[:,d])/Z

    Weights = np.empty((N,2**D))
    for i in xrange(2**D):
        b_str = np.binary_repr(i,width=D)
        b = np.array([int(x) for x in b_str],dtype=bool)
        vert = np.empty(2*D,dtype=bool)
        vert[::2] = b
        vert[1::2] = 1 - b
        Weights[:,i] = np.prod(Coef[:,vert],axis=1)
    return Weights
    
def test_interp():
    x = np.array([[0.2,0.1]])
    low = np.array([0,0])
    high = np.array([1,1])
    print trial_interpolate(x,low,high)

test_2d()
