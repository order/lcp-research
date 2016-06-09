import numpy as np
import scipy as sp
import scipy.sparse as sps
import itertools

import indexer
import discretize

class RegularGridInterpolator(object):
    def __init__(self,grid_desc):
        self.D = len(grid_desc)
        self.grid_desc = grid_desc # List of (low,high,num) triples

        # Number of cutpoints along each dimension
        self.lengths = np.array([n+1 for (l,h,n)
                                 in self.grid_desc])
        for l in self.lengths:
            assert(l >= 1)

        # Find the low points in the grid
        self.low = np.array([l for (l,h,n) in self.grid_desc])

        # Get the spacing for each dimension
        self.delta = np.array([float(h-l) / float(n)
                               for (l,h,n) in self.grid_desc])

        # Initialize the indexer
        self.indexer = indexer.Indexer(self.lengths)

        # One for every cutpoint, plus an oob
        self.num_nodes = self.indexer.max_index+1

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
            (low,high,n) = self.grid_desc[d]
            # Linearly transform the data so [low,high)
            # will be in [0,n)
            transform = n * (points[:,d] - low) / (high - low)
            Coords[:,d] = np.floor(transform)

            # Fuzz top boundary to get [low,high]
            fuzz_mask = np.logical_and(high <= points[:,d],
                                     points[:,d] < high + self.fuzz)
            Coords[fuzz_mask,d] = n-1         
        return Coords
        
    def points_to_index_distributions(self,points):
        (N,D) = points.shape
        assert(D == self.D)
        
        coords = self.to_cell_coord(points)
        indices = self.indexer.coords_to_indices(coords)
        # Index of low cutpoint in cell

        #dist = np.empty((N,D))
        #for d in xrange(D):
        #    (l,h,n) = self.grid_desc[d]
        #    low_coord = l + coords[:,d]*self.delta[d]
        #    dist[:,d] = (points[:,d] - low_coord) / self.delta[d]
        dist = (points - self.low - coords*self.delta) / self.delta

        weights = np.empty((N,2**D))
        for (i,diff) in enumerate(itertools.product([0,1],
                                                    repeat=D)):
            mask = np.array(diff,dtype=bool)
            weights[:,i] = np.product(dist[:,mask],axis=1)\
                           * np.product(1 - dist[:,~mask],axis=1)

        cell_shift = self.indexer.cell_shift() # co-cell neighbor shift
        vertices = indices[:,np.newaxis] + cell_shift[np.newaxis,:]
        assert((N,2**D) == vertices.shape)

        # Set up some masks and indices
        big_physical = self.indexer.physical_max_index
        big_oob = self.indexer.max_index
       
        oob_mask = (indices > big_physical)
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
        rows[B:] = indices[oob_idx]
        data[B:] = np.ones(num_oob)
        
        M = self.num_nodes
        return sps.coo_matrix((data,(rows,cols)),shape=(M,N))    

    def get_cutpoints(self):
        linspaces = [np.linspace(l,h,n+1) for (l,h,n)
                     in self.grid_desc]
        N = self.num_nodes
        n = self.indexer.physical_max_index+1
        D = self.D
        points = np.empty((N,D))
        points[:n,:] = discretize.make_points(linspaces)
        points[n:,:] = np.nan
        return points

    def has_point(self,target):
        (N,) = target.shape
        for (i,(l,h,n)) in enumerate(self.grid_desc):
            skip = (target[i] - l) / n
            if (skip % 1) > 1e-15:
                return False
        return True
        
    
    def indices_to_points(self,indices):
        (N,) = indices.shape
        D = self.D

        # Convert to coords
        coords = self.indexer.indices_to_coords(indices)
        assert((N,D) == coords.shape)

        # (D,) + (N,D) * (D,) ; should be (N,D) at end
        points = self.low + coords * self.delta
        assert((N,D) == points.shape)

        return points
        
