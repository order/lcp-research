import numpy as np
import scipy.sparse as sps

from rectilinear_indexer import Indexer
from coord import OutOfBounds, Coordinates

from utils import is_sorted,is_vect,is_mat,row_vect,col_vect        

##############################
# ABSTRACT RECTILINEAR GRID #
##############################
class Grid(object):
    def points_to_cell_indices(self,points):
        """
        Main function: convert points to cell indices.
        The cells mapped to are the cells that contain them.
        Points are (N,D) arrays.
        """
        raise NotImplementedError()
    def points_to_cell_coords(self,points):
        """
        Convert points to cell coords
        Points are (N,D) arrays.
        Coords are internal data structures containing the grid
        coordinates for the cell, and out of bounds information
        """
        raise NotImplementedError()

    def cell_indices_to_mid_points(self,cell_indices):
        """
        Map indices to cell midpoint
        o - o
        | x |
        o - o
        """
        raise NotImplementedError()  
    def cell_indices_to_low_points(self,cell_indices):
        """
        Map indices to cell lowpoint
        o - o
        |   |
        x - o 
        """
        raise NotImplementedError()

    def node_indices_to_node_points(self,node_indices):
        raise NotImplementedError()

    def cell_indices_to_vertex_indices(self,cell_indices):
        """
        Map cell indices to the vertex indices of the cell surrounding them.
        Vertex indices are node indices.
        """
        raise NotImplementedError()
    def cell_coords_to_vertex_indices(self,cell_coords):        
        raise NotImplementedError()
    

    def point_to_low_vertex_rel_distance(self,points):
        """
        Map points to the relative distance (i.e. scale so the cell is a
        unit hypercube) from them to the low point in the cell.
        """
        raise NotImplementedError()

    """
    Gridded space is an axis-aligned rectangle
    """
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

    def get_num_total_cells(self):
        # Total number (scalar), include oob
        return self.cell_indexer.max_index+1
    
    def get_num_spatial_cells(self):
        # Exclude oob
        return self.cell_indexer.spatial_max_index+1

    def get_num_total_nodes(self):
        # Total number (scalar), include oob
        return self.nodes_indexer.max_index+1
    
    def get_num_spatial_nodes(self):
        # Exclude oob
        return self.nodes_indexer.spatial_max_index+1
    
    def get_num_oob(self):
        return self.num_nodes() - self.num_spatial_nodes()
    
    def get_oob_range(self):
        return xrange(self.num_spatial_nodes(),self.num_nodes())


##############################
# REGULAR RECTILINEAR GRID #
##############################
class RegularGrid(Grid):
    def __init__(self,grid_desc):
        assert isinstance(grid_desc,(list,tuple))
        for gd in grid_desc:
            assert isinstance(gd,(list,tuple))
            assert 3 == len(gd)        
        self.dim = len(grid_desc)
        self.grid_desc = grid_desc # List of (low,high,num) triples

        (low,hi,num_cells) = zip(*self.grid_desc)
        self.lower_bound = np.array(low,dtype=np.double)
        self.upper_bound = np.array(hi,dtype=np.double)
        self.num_cells = np.array(num_cells,dtype=np.integer)
        assert not np.any(self.num_cells <= 0)
        self.num_nodes = self.num_cells + 1

        # Cell dimensions
        self.delta = (self.upper_bound - self.lower_bound)
        self.delta /= self.num_cells.astype(np.double)

        # Initialize the indexer
        self.cell_indexer = Indexer(self.num_cells)
        self.node_indexer = Indexer(self.num_nodes)

        # Fuzz to convert [low,high) to [low,high]
        self.fuzz = 1e-15

    def points_to_cell_coords(self,points):
        """
        Figure out where points are. Returns the cell coordinate.
        """
        assert is_mat(points) 
        (N,D) = points.shape
        assert D == self.dim
        
        # Get the OOB info
        oob = OutOfBounds(self,points)
        
        coords = np.empty((N,D))
        for d in xrange(D):
            (low,high,num_cells) = self.grid_desc[d]
            # Transform: [low,high) |-> [0,n)
            transform = num_cells * (points[:,d] - low) / (high - low)
            coords[:,d] = np.floor(transform + self.fuzz)
            # Add a little fuzz to make sure stuff on the boundary is
            # mapped correctly

            # Fuzz top boundary to get [low,high]
            fuzz_mask = np.logical_and(high <= points[:,d],
                                     points[:,d] < high + 2*self.fuzz)
            coords[fuzz_mask,d] = num_cells - 1
            # Counts things just a littttle bit greater than last cell
            # boundary as part of the last cell            
        coords[~oob.mask,:] = np.nan
        
        return Coordinates(coords,oob)
    
    def points_to_cell_indices(self,points):
        assert is_mat(points)
        (N,D) = points.shape
        
        cell_coords = self.points_to_cell_coords(points)
        assert isinstance(cell_coords,Coordinates)
        assert (N,D) == cell_coords.shape
        
        cell_indices = self.cell_indexer.coords_to_indices(cell_coords)
        assert is_vect(cell_indices)
        assert (N,) == cell_indices.shape
        
        return cell_indices

    def cell_indices_to_mid_points(self,cell_indices):
        assert is_vect(cell_indices)

        low_points = cell_indices_to_low_points(self,cell_indices)
        mid_points = low_points + row_vect(0.5 * self.delta)
        assert is_mat(mid_points)
        assert mid_points.shape[0] == cell_indices.shape[0]
        
        return mid_points

    def cell_indices_to_low_points(self,cell_indices):
        assert is_vect(cell_indices)
        
        cell_coords = self.cell_indexer.indices_to_coords(cell_indices)
        assert isinstance(cell_coords,Coordinates)
        
        low_points = self.cell_coords_to_low_points(cell_coords)
        assert is_mat(low_points)
        assert cell_coords.shape == low_points.shape

        return low_points
        
        
    def cell_coords_to_low_points(self,cell_coords):
        assert isinstance(cell_coords,Coordinates)
        assert self.dim == cell_coords.dim
        
        C = cell_coords.coords
        oob = cell_coords.oob
        low_points = row_vect(self.lower_bound) + C * row_vect(self.delta)

        assert is_mat(low_points)
        assert np.all(low_points[oob.mask,:] == np.nan)
        assert cell_coords.shape == low_points.shape
        return low_points
    
    def node_indices_to_node_points(self,node_indices):
        assert is_vect(node_indices)
        (N,) = node_indices.shape
        
        node_coords = self.node_indexer.indices_to_coords(node_indices)
        assert isinstance(node_coords,Coordinates)
        
        C = node_coords.coords
        oob = node_coords.oob
        node_points = row_vect(self.lower_bound) + C * row_vect(self.delta)
        assert is_mat(node_points)
        assert np.all(node_points[oob.mask,:] == np.nan)
        assert cell_coords.shape == node_points.shape
        
        return node_points

    def cell_indices_to_vertex_indices(self,cell_indices):
        assert is_vect(cell_indices)
        
        cell_coords = self.cell_indexer.indices_to_coords(cell_indices)
        assert isinstance(cell_coords,Coordinates)
        
        vertex_indices = self.cell_coords_to_vertex_indices(cell_coords)
        assert is_mat(vertex_indices) # (N x 2**D) matrix
        
        return vertex_indices
        
    def cell_coords_to_vertex_indices(self,cell_coords):
        assert isinstance(cell_coords,Coordinates)
        (N,D) = cell_coords.shape
        assert self.dim == D


        """
        The low node index in the cell has the same coords in node-land
        as the cell in cell-land:
         |   |
        -o - o-
         | x |
        -x - o-
         |   |
        """        
        low_vertex = self.node_indexer.coords_to_indices(cell_coords)

        # Array of index offsets to reach every vertex in cell
        shift = self.node_indexer.cell_shift()
        assert (2**D,) == shift.shape
        
        vertices = col_vect(low_vertex) + row_vect(shift)
        assert (N,2**D) == vertices.shape

        """
        Handle out of bound nodes. There is a constant offset for 
        converting cell oob indices to node oob indices.
        Also the difference between max spatial indices.
        """
        oob = cell_coords.oob
        if oob.has_oob():
            # Figure out the right oob node
            oob_indices = low_vertex[oob.mask]
            offset = self.node_indexer.get_num_spatial_nodes() \
                     - self.cell_indexer.get_num_spatial_nodes()
            vertices[oob.mask,0] = col_vect(oob_indices) + offset
            vertices[oob.mask,1:] = np.nan
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

