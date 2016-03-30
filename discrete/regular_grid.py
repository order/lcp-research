import numpy as np

##########################
# REGULAR GRID PARTITION

class RegularGridPartition(object):
    """
    Defines an grid split based on regular cutpoints
    """
    def __init__(self,grid_desc):
        D = len(grid_desc)
        self.grid_desc = grid_desc # List of (low,high,num) triples

        self.dim = D
        self.lengths = np.array([n for (l,h,n) in self.grid_desc])
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

####################################
# REGULAR GRID DISCRETIZER

class RegularGridDiscretizer(discretize.Discretizer):
    def __init__(self,grid_desc):
        self.partition = RegularGridPartition(grid_desc)
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
