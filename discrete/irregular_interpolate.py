import numpy as np
import scipy as sp
import scipy.sparse as sps
import itertools

import indexer
import discretize

class IrregularGridInterpolator(object):
    def __init__(self,grids):
        self.D = len(grids)
        self.grids = grids # Explicit grids

        # Number of cutpoints along each dimension
        self.lengths = np.array([len(g) for g in self.grids])
        for l in self.lengths:
            assert(l > 1)

        # One for every cutpoint, plus an oob
        self.num_nodes = np.prod(self.lengths) + 1

        # Initialize the indexer
        self.indexer = indexer.Indexer(self.lengths)

        # Fuzz to convert [low,high) to [low,high]
        self.fuzz = 1e-15
        
    def to_cell_coord(self,points):
        """
        Figure out which cell each point is in
        """
        (N,D) = points.shape
        assert(D == self.D)
        
        Coords = np.empty((N,D))
        for d in xrange(D):
            K = self.lengths[d]
            coord = np.searchsorted(self.grids[d],
                                    points[:,d],
                                    side='right') - 1
            Coords[:,d] = coord

            # Fuzz to deal with difference between [low,hi) and [low,hi]
            hi = self.grids[d][-1]
            fuzz_mask = np.where(np.logical_and(points[:,d] >= hi,
                                              points[:,d] < hi+self.fuzz))
            Coords[fuzz_mask,d] = K-2                       

            (left,right) = (self.grids[d][0],self.grids[d][-1])
            oob_mask = np.logical_or(points[:,d] < left,
                                     points[:,d] >= right + self.fuzz)
            Coords[oob_mask,d] = np.nan         
        return Coords
        
    def points_to_index_distribution(self,points):
        (N,D) = points.shape
        assert(D == self.D)
        
        coords = self.to_cell_coord(points)
        indices = self.indexer.coords_to_indices(coords)
        # Index of low cutpoint in cell

        nan_mask = np.any(np.isnan(coords),axis=1)
        dist = np.empty((N,D))
        for d in xrange(D):
            idx = coords[~nan_mask,d].astype('i')
            low_point= self.grids[d][idx]
            hi_point = self.grids[d][idx + 1]
            delta = (hi_point - low_point)
            dist[~nan_mask,d] = (points[~nan_mask,d] - low_point) / delta
        dist[nan_mask,:] = np.nan

        weights = np.empty((N,2**D))
        for (i,diff) in enumerate(itertools.product([0,1],repeat=D)):
            mask = np.array(diff,dtype=bool)
            weights[:,i] = np.product(dist[:,mask],axis=1)\
                           * np.product(1 - dist[:,~mask],axis=1)

        cell_shift = self.indexer.cell_shift() # co-cell neighbor shift
        vertices = indices[:,np.newaxis] + cell_shift[np.newaxis,:]
        assert((N,2**D) == vertices.shape)

        # Set up some masks and indices
        oob_node = self.indexer.max_index
        oob_mask = (indices == oob_node)
        normal_idx = np.arange(N)[~oob_mask]
        oob_idx = np.arange(N)[oob_mask]
        num_oob = np.sum(oob_mask)
        num_norm = N - num_oob

        B = num_norm*(2**D) # Space for normal points
        L = B + num_oob # Add on space for oob nodes
        cols = np.empty(L)
        rows = np.empty(L)
        data = np.empty(L)

        # Add normal weights
        cols[:B] = (np.tile(normal_idx,(2**D,1)).T).flatten()
        rows[:B] = (vertices[~oob_mask,:]).flatten()
        data[:B] = (weights[~oob_mask,:]).flatten()

        # Route all oob points to oob node
        cols[B:] = oob_idx
        rows[B:] = np.full(num_oob,oob_node)
        data[B:] = np.ones(num_oob)
        
        M = self.num_nodes
        return sps.coo_matrix((data,(rows,cols)),shape=(M,N))    

    def get_cutpoints(self):
        N = self.num_nodes
        D = self.D
        points = np.empty((N,D))
        points[:-1,:] = discretize.make_points(self.grids)
        points[-1,:] = np.nan
        return points
        
    
    def indices_to_points(self,indices):
        (N,) = indices.shape
        D = self.D

        # Convert to coords
        coords = self.indexer.indices_to_coords(indices)
        assert((N,D) == coords.shape)

        # (D,) + (N,D) * (D,) ; should be (N,D) at end
        points = np.empty((N,D))
        for d in xrange(D):
            points[:,d] = self.grid[d][coords[:,d]]

        return points
        
