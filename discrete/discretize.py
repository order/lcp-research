import numpy as np
import scipy.sparse as sps

class Discretizer(object):
    def points_to_index_distributions(self,points):
        """
        Takes in (n,d) points,
        Returns (n,N) sparse row-stochastic matrix indicating index
        distribution
        """
        raise NotImplementedError()
    def indices_to_points(self,indices):
        """
        Takes in (n,) indices,
        Returns (n,d) matrix indicating canonical points associated
        with each
        """
        raise NotImplementedError()

    def get_cutpoints(self):
        raise NotImplementedError()

    def has_point(self,target):
        (N,) = target.shape
        for (i,(l,h,n)) in enumerate(self.grid_desc):
            skip = (target[i] - l) / n
            if (skip % 1) > 1e-15:
                return False
        return True
        

class TrivialDiscretizer(object):
    def __init__(self,num_nodes):
        self.num_nodes = num_nodes
    def points_to_index_distributions(self,points):
        (N,D) = points.shape
        assert(D == 1)
        assert(not np.any(points < 0))
        assert(not np.any(points >= self.num_nodes))
        data = np.ones(N)
        row = points.flatten().astype('i')
        col = np.arange(N)
        return sps.coo_matrix((data,(row,col)),
                              shape=(self.num_nodes,N))
    def indicies_to_points(self,indices):
        (N,) = indicies.shape
        return np.array(indices).reshape((N,1))
    def get_cutpoints(self):
        N = self.num_nodes
        return np.arange(N)[:,np.newaxis]
    def has_point(self,target):
        assert((1,) == target)
        return 0 <= target[0] < self.num_nodes\
           and (target[0] % 1) < 1e-15
    

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

def partition_samples(S,K):
    """
    Parition samples into D^K partitions by order statistic
    """
    (N,D) = S.shape

    percent = np.linspace(0,100,K+1) # Percentiles
    Cutpoints = []

    for d in xrange(D):
        cuts = np.empty(K+1)
        for k in xrange(K+1):
            p = np.percentile(S[:,d],percent[k])
            cuts[k] = p
        Cutpoints.append(cuts)
    return Cutpoints

def weighted_centroid(S,discretizer,indexer):
    (N,D) = S.shape
    M = discretizer.num_cells
    coords = discretizer.to_cell_coord(S)
    indices = indexer.coords_to_indices(coords)

    unique = np.unique(indices)

    centroid = np.full((M,D),np.nan,dtype=np.double)
    for idx in unique:
        mask = (indices == idx)
        centroid[idx,:] = np.mean(S[mask,:],axis=0)

    return centroid
    
def grid_desc_to_lhn(grid_desc):
    (low,high,num) = zip(*grid_desc)
