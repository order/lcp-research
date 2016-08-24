import numpy as np
import scipy.sparse as sps

from rectilinear_indexer import Indexer

from utils import is_sorted,row_vect,col_vect

############
# OOB DATA #
############

class OutOfBoundsData(object):
    def __init__(self,grid,points):
        (N,D) = points.shape
        self.dim = D
        self.num = N
        
        low = grid.get_lower_boundary()
        high = grid.get_upper_boundary()


        # What boundary is violated;
        # -1 lower boundary,
        # +1 upper boundary
        self.data = 1.0*sps.csc_matrix(points > high) \
               - 1.0*sps.csc_matrix(points < row_vect(low))
        
        # Points that violate some boundary
        self.rows = (self.data.tocoo()).row

        # Mask of same
        self.mask = np.zeros(N,dtype=bool)
        self.mask[self.rows] = True

        # Pre-offset oob node or cell indices
        self.indices = self.find_oob_index()
        
    def find_oob_index(self):
        """
        Linearizes the oob information
        """
        (N,) = self.rows.shape
        indices = np.empty(N)
        # Reverse order, larger overwrites smaller
        for d in xrange(self.dim-1,-1,-1):
            # All points that are too small in this dimension
            small = (self.data.getcol(d) == -1).nonzero()[0]
            indices[small] = 2*d

            # ALl points that are too large in this dimension
            large = (self.data.getcol(d) == 1).nonzero()[0]
            indices[large] = 2*d + 1
        return indices
            
                

##############################
# ABSTRACT RECTILINEAR GRID #
##############################
class Grid(object):
    # Main functions: convert points to cell coords and indices
    def points_to_cell_coords(self,points,oob):
        raise NotImplementedError()
    def points_to_cell_indices(self,points,oob):
        raise NotImplementedError()

    # Map indices to cell midpoint
    # o - o
    # | x |
    # o - o
    def cell_indices_to_mid_points(self,cell_indices):
        raise NotImplementedError()
    # Map indices to cell lowpoint
    # o - o
    # |   |
    # x - o   
    def cell_indices_to_low_points(self,cell_indices):
        raise NotImplementedError()

    def node_indices_to_node_points(self,node_indices):
        raise NotImplementedError()

    # Map cell indices to the vertex indices of the cell surrounding them
    def cell_indices_to_vertex_indices(self,cell_indices):
        raise NotImplementedError()
    def cell_coords_to_vertex_indices(self,cell_coords):
        raise NotImplementedError()
    # Map points to the relative distance (i.e. scale so the cell is a
    # unit hypercube) from them to the low point in the cell.
    def point_to_low_vertex_rel_distance(self,points):
        raise NotImplementedError()
    
    # Overall space is an axis-aligned rectangle
    def get_lower_boundary(self):
        return self.lower_bound    
    def get_upper_boundary(self):
        return self.upper_bound
    def get_dim(self):
        return self.dim

    def are_points_oob(self,points):
        raise NotImplementedError()

    def get_cell_indexer(self):
        return self.cell_indexer

    def get_node_indexer(self):
        return self.node_indexer

    def get_num_cells(self):
        # Total number (scalar), include oob
        return self.cell_indexer.max_index+1
    
    def get_num_real_cells(self):
        # Exclude oob
        return self.cell_indexer.physical_max_index+1

    def get_num_nodes(self):
        # Total number (scalar), include oob
        return self.nodes_indexer.max_index+1
    
    def get_num_real_nodes(self):
        # Exclude oob
        return self.nodes_indexer.physical_max_index+1
    
    def get_num_oob(self):
        return self.num_nodes() - self.num_real_nodes()
    
    def get_oob_range(self):
        return xrange(self.num_real_nodes(),self.num_nodes())


##############################
# REGULAR RECTILINEAR GRID #
##############################
class RegularGrid(Grid):
    def __init__(self,grid_desc):
        self.dim = len(grid_desc)
        self.grid_desc = grid_desc # List of (low,high,num) triples

        # Number of cutpoints along each dimension
        (low,hi,num_cells) = zip(*self.grid_desc)
        self.lower_bound = np.array(low,dtype=np.double)
        self.upper_bound = np.array(hi,dtype=np.double)
        self.num_cells = np.array(num_cells)
        assert not np.any(self.num_cells == 0)
        self.num_nodes = self.num_cells + 1

        # Cell dimensions
        self.delta = (self.upper_bound - self.lower_bound)
        self.delta /= self.num_cells.astype(np.double)

        # Initialize the indexer
        self.cell_indexer = Indexer(self.num_cells)
        self.node_indexer = Indexer(self.num_nodes)

        # Fuzz to convert [low,high) to [low,high]
        self.fuzz = 1e-15

    def points_to_cell_coords(self,points,oob=None):
        """
        Figure out where points are. Returns the cell coordinate.
        """

        if oob is None:
            oob = OutOfBoundsData(self,points)
        
        (N,D) = points.shape
        assert D == self.dim
        
        Coords = np.empty((N,D))
        for d in xrange(D):
            (low,high,num_cells) = self.grid_desc[d]
            # Transform: [low,high) |-> [0,n)
            transform = num_cells * (points[:,d] - low) / (high - low)
            Coords[:,d] = np.floor(transform + self.fuzz)
            # Add a little fuzz to make sure stuff on the boundary is
            # mapped correctly

            # Fuzz top boundary to get [low,high]
            fuzz_mask = np.logical_and(high <= points[:,d],
                                     points[:,d] < high + 2*self.fuzz)
            Coords[fuzz_mask,d] = num_cells - 1
            # Counts things just a littttle bit greater than last cell
            # boundary as part of the last cell
            
        Coords[~oob.mask,:] = np.nan
        return Coords
    
    def points_to_cell_indices(self,points,oob=None):
        if oob is None:
            oob = OutOfBoundsData(self,points)
        cell_coords = self.points_to_cell_coords(points,oob)
        cell_indices = self.cell_indexer.coords_to_indices(cell_coords,oob)
        return cell_indices

    def cell_indices_to_mid_points(self,cell_indices):
        low_points = cell_indices_to_low_points(self,cell_indices)
        return low_points + row_vect(0.5 * self.delta)

    def cell_indices_to_low_points(self,cell_indices):
        assert 1 == cell_indices.ndim        
        cell_coords = self.cell_indexer.indices_to_coords(cell_indices)
        return self.cell_coords_to_low_points(cell_coords)
        
    def cell_coords_to_low_points(self,cell_coords):
        assert 2 == cell_coords.ndim
        (N,D) = cell_coords.shape
        assert self.dim == D
        points = self.lower_bound + cell_coords * row_vect(self.delta)
        assert (N,D) == points.shape
        return points

    def node_indices_to_node_points(self,node_indices):
        node_coords = self.node_indexer.indices_to_coords(node_indices)
        return row_vect(self.lower_bound) + node_coords * row_vect(self.delta)

    def cell_indices_to_vertex_indices(self,cell_indices):
        cell_coords = self.cell_indexer.indices_to_coords(cell_indices)
        return self.cell_coords_to_vertex_indices(cell_coords)
        
    def cell_coords_to_vertex_indices(self,cell_coords):
        (N,D) = cell_coords.shape
        assert self.dim == D

        oob_mask = self.cell_indexer.are_coords_oob(cell_coords)
        
        # The low node index in the cell has the same coords in node-land
        # as the cell in cell-land
        low_vertex = self.node_indexer.coords_to_indices(cell_coords)

        shift = self.node_indexer.cell_shift()
        assert (2**D,) == shift.shape
        
        vertices = col_vect(low_vertex) + row_vect(shift)        
        
        if oob_mask.sum() > 0:
            # Figure out the right oob node
            oob_coords = cell_coords[oob_mask,:]
            cell_oob = self.cell_indexer.coords_to_indices(oob_coords)
            node_offset = self.node_indexer.get_num_nodes() \
                          - self.cell_indexer.get_num_nodes()
            vertices[oob_mask,:] = col_vect(cell_oob) + node_offset
        return vertices        

    def point_to_low_vertex_rel_distance(self,points,cell_indices):
        raise NotImplementedError()

    def are_points_oob(self,points):
        """
        Check if points are out-of-bounds
        """
        (N,D) = points.shape
        assert D == self.dim

        L = np.any(points < row_vect(self.lower_bound),axis=1)
        U = np.any(points > row_vect(self.upper_bound) + self.fuzz,axis=1)
        assert (N,) == L.shape
        assert (N,) == U.shape

        return np.logical_or(L,U)

##############################
# IRREGULAR RECTILINEAR GRID #
##############################
class IrregularGrid(Grid):
    """
    Rectilinear grid from irregular, but sorted, list of node locations
    """
    def __init__(self,node_lists):
        self.dim = len(node_lists)
        self.node_lists = np.array(node_lists)
        # List of np.ndarray cutpoint locations

        for nl in node_lists:
            assert nl.ndim == 1 # 1D array
            assert nl.size >= 2 # At least two nodes
            assert is_sorted(nl)
        
        # Number of cutpoints along each dimension
        desc = [(nl[0],nl[-1],nl.size) for nl in node_lists]
        (low,hi,num) = zip(*desc)
        self.lower_bound = np.array(low)
        self.upper_bound = np.array(hi)
        self.num_nodes = np.array(num)
        
        self.num_cells = self.num_nodes - 1

        # Initialize the indexer
        self.indexer = Indexer(self.num_nodes)

        # Fuzz to convert [low,high) to [low,high]
        self.fuzz = 1e-12
    
    def points_to_cell_coords(self,points):
        (N,D) = points.shape
        assert D == self.dim

        Coords = np.empty((N,D))
        for d in xrange(D):
            # Find the correct position in the dth node list
            coord = np.searchsorted(self.node_lists[d],
                                    points[:,d],
                                    side='right') - 1
            # The 'right' is important if points are exactly on the node
            assert (N,) == coord.shape
            Coords[:,d] = coord

            # Include the upper boundary
            ub = self.upper_bound[d]
            hi_cell = self.num_cells[d] - 1
            fuzz_mask = np.logical_and(points[:,d] >= ub,
                                       points[:,d] < ub + self.fuzz)
            Coords[fuzz_mask,d] = hi_cell

            # Indexer will take care of mapping to correct OOB node
            #lb = self.lower_bound[d]
            #oob_mask = np.logical_or(points[:,d] < lb,
            #                         points[:,d] >= ub+self.fuzz)
            #Coords[oob_mask,d] = np.nan
        return Coords

    def points_to_indices(self,points):
        coords = self.points_to_cell_coords(points)
        return self.indexer.coords_to_indices(coords)

    def indices_to_lowest_points(self,indices):
        assert 1 == indices.ndim
        
        coords = self.indexer.indices_to_coords(indices)
        return self.coords_to_lowest_points(coords)
        
    def coords_to_lowest_points(self,coords):
        assert 2 == coords.ndim
        (N,D) = coords.shape
        assert self.dim == D

        points = np.empty((N,D))
        for d in xrange(D):
            points[:,d] = self.node_lists[d,coords[:,d]]

        return points

