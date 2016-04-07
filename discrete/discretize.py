import numpy as np

class Discretizer(object):
    def points_to_indices(self,points):
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

    centroid = np.full((M,D),np.nan)
    for idx in unique:
        mask = (indices == idx)
        centroid[idx,:] = np.mean(S[mask,:],axis=0)

    return centroid
    


