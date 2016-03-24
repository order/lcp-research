import numpy as np

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
        
        self.num_cpts_per_dim = [len(x) for x in cpts]
        self.lens = [len(x)-1 for x in cpts]
        self.num_cells = np.prod(self.lens)
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
            K = part.lens[d]
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
