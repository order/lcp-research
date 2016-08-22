import numpy as np
import scipy.sparse as sps

class Discretizer(object):
    
    def locate(self,points):
        """
        Takes in (n,d) points,
        LOCATES the cells that they are in,
        Returns the appropriated distribution
        node indices
        """
        raise NotImplementedError()

#######################
# TABULAR DISCRETIZER #
#######################

class TabularDiscretizer(Discretizer):
    def __init__(self,grid):
        self.grid = grid
        self.dim = grid.dim
        
        self.node_indexer = grid.get_indexer()
        lens = self.node_indexer.lens
        assert np.all(lens >= 2) # 2 nodes, one cell
        self.num_nodes = lens
        self.num_cells = lens - 1
        self.cell_indexer = Indexer(self.num_cells)

        self.oob_offset = self.node_indexer.physical_max_index \
                          - self.cell_indexer.physical_max_index
        

    def remap_node_indices_to_cell_indices(self,node_indices):
        """
        Map least indexed vertex to cell index
        
        6 - 7 - 8
        | 3 | 4 |
        3 - 4 - 5
        | 0 | 1 |
        0 - 1 - 2

            |
            v

        6 - 7 - 8
        | 2 | 3 |
        3 - 4 - 5
        | 0 | 1 |
        0 - 1 - 2
        """
        node_oob_mask = self.node_indexer.is_oob(node_indices)
        node_coords = self.node_indexer.indices_to_coords(node_indices)

        node_coords[node_oob_mask,:] = -1
        # Nothing has final coord
        for d in xrange(self.dim):
            assert(node_coords[node_oob_mask,d] < self.num_nodes-1)

        cell_indices = self.cell_indexer.coords_to_indices(node_coords)
        cell_oob_mask = self.cell_indexer.is_oob(cell_indices)
        assert(np.all(cell_oob_mask == node_oob_mask))

        cell_indices[node_oob_mask] = node_indices - self.oob_offset
        
        return cell_indices      
    
        
    def locate(self,points):
        """
        The location distribution for a point is the 
        index of the cell that it falls in.

        6 - 7 - 8
        | 2 | 3 |
        3 - 4 - 5
        | 0 | 1 |
        0 - 1 - 2
        
        """
        (N,D) = points.shape
        assert (D == self.grid.get_dim())

 
        indices = self.grid.points_to_indices(points)
        assert(not np.any(np.isnan(indices)))
        
        cols = np.arange(N)
        rows = indices
        data = np.ones(N,dtype=np.double)

        M = self.grid.get_num_nodes()
        return sps.coo_matrix((data,(rows,cols)),shape=(M,N))

#############################
# MULTILINEAR INTERPOLATION #
#############################

def MultilinearInterpolation(object):
    def __init__(self,grid):
        self.grid = grid

    def convert_to_sparse_matrix(self,indices,vertices,weights):
        (N,) = indices.shape
        D = self.grid.get_dim()

        indexer = self.grid.get_indexer()
        oob_mask = indexer.is_oob(indices)
        num_oob = np.sum(oob_mask)
        num_normal = N = num_oob
        normal_idx = np.arange(N)[~oob_mask]
        oob_idx = np.arange(N)[oob_mask]
        
        m = num_norm*(2**D) # Space for normal points
        M = m + num_oob # Add on space for oob nodes
        cols = np.empty(M)
        rows = np.empty(M)
        data = np.empty(M)

        # Add normal weights
        cols[:m] = (np.tile(normal_idx,(2**D,1)).T).flatten()
        rows[:m] = (vertices[~oob_mask,:]).flatten()
        data[:m] = (weights[~oob_mask,:]).flatten()

        # Route all oob points to oob node
        cols[B:] = oob_idx
        rows[B:] = indices[oob_idx]
        data[B:] = np.ones(num_oob)

        NN = self.grid.get_num_nodes()
        return sps.coo_matrix((data,(rows,cols)),shape=(NN,N))           
            
    def locate(self,points):
        (N,D) = points.shape
        assert D == self.grid.get_dim()

        # Get indices
        indices = points_to_indices(points) # Compute once
        
        dist = self.grid.get_rel_distance_to_low(points,indices)
        assert (N,D) == dist

        vertices = self.grid.get_neighbors(indices)
        assert (N,2**D) == vertices.shape
        
        # Calculate multilinear interp weights from distances
        weights = np.empty((N,2**D))
        for (i,diff) in enumerate(itertools.product([0,1],
                                                    repeat=D)):
            mask = np.array(diff,dtype=bool)
            weights[:,i] = np.product(dist[:,mask],axis=1)\
                           * np.product(1 - dist[:,~mask],axis=1)

        return convert_to_sparse_matrix(self,indices,vertices,weights)
        

#######################################
# AUX functions    

def is_int(x):
    f = np.mod(x,1.0)
    mask = ~np.isnan(x)

    return np.all(f[mask]<1e-9)


def make_points(gens,ret_mesh=False,order='C'):
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
    if ret_mesh:
        return points,meshes
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
