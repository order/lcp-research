import numpy as np
import scipy.sparse
import itertools
import operator
import bisect

class NodeMapper(object):
    """
    Abstract class defining state-to-node mappings. 
    For example, basic grid mappings or fixing OOB states.
    """
    def covers(self,state):
        """
        Checks if a node mapper is responsible for mapping state
        """
        raise NotImplementedError()
    def states_to_node_dists(self,states,**kwargs):
        """
        Takes an arbitrary Nxd ndarray and maps it to
        each of the rows to a distribution over nodes
        """
        raise NotImplementedError()
    def nodes_to_states(self,nodes):
        """
        Maps each node to its canonical state. If the node
        is a abstract one, like "out-of-bounds", then the
        state should be NaN.
        """
        raise NotImplementedError()  
        
    def get_num_nodes(self):
        """
        Get the number of nodes that the node mapper is
        responsible for
        """
        raise NotImplementedError()        
        
    def get_node_ids(self):
        """
        Get a list/iterator of node ids
        """
        raise NotImplementedError()

    def get_node_states(self):
        """
        Get the canonical state for every node mapper is responsible for
        """
        raise NotImplementedError()
        
    def get_dimension(self):
        """
        Return the dimension of the state-space
        """
        raise NotImplementedError()
        
class BasicNodeMapper(NodeMapper):
    def states_to_transition_matrix(self,states,**kwargs):
        """
        Build a transition matrix from the states to nodes.
        Useful in building MDP transition matrices
        
        Hopefully will be more efficient than building lists of NodeDists
        """
        raise NotImplementedError()
     
class NodeDist(object):
    """
    Stores information for a distribution over nodes
    """
    def __init__(self,*vargs):
        """
        Initializes NodeDist from either 
        """
        self.dist = {}
        if len(vargs) == 2:
            self.dist[vargs[0]] = vargs[1]
        else:
            assert(len(vargs) == 0)
        
    def add(self,node_id,weight):
            if node_id in self.dist:
                raise KeyError('{0} already in dist table'.format(node_id))
            self.dist[node_id] = weight
            
    def keys(self):
        return self.dist.keys()
        
    def items(self):
        return self.dist.items()
        
    def normalize(self):
        agg = 0.0
        for v in self.dist.values():
            agg += v
        for k in self.dist:
            self.dist[k] /= agg
            
    def verify(self):
        agg = 0.0
        for v in self.dist.values():
            agg += v
        if abs(agg - 1.0) > 1e-8:
            raise AssertionError('Node distribution sums to {0}'.format(agg))
            
    def get_unique_node_id(self):
        assert(len(self.dist) == 1)
        return self.dist.keys()[0]
            
    def __getitem__(self,index):
        return self.dist[index]
        
    def __len__(self):
        return len(self.dist)
            
    def __str__(self):
        return '{' + ', '.join(map(lambda x: '{0}:{1:.3f}'.format(x[0],x[1]), sorted(self.dist.items()))) + '}'
        
class OOBSinkNodeMapper(NodeMapper):
    """
    Ensures that a state doesn't exceed an (axis-aligned) boundary by sending to sink state
    """
    def __init__(self,dim,low,high,sink_node):
        """
        low <= state[dim] <= high; otherwise map state to sink_node
        """
        assert(low < high)
        self.dim = dim
        self.low = low 
        self.high = high
        self.sink_node = sink_node # this is the node id for the sink state
        
    def covers(self,state):
        return (low <= state[self.dim] <= high)
        
    def states_to_node_dists(self, states, ignore):
        """
        Assumes states to be an N x d np.array
        This is a partial mapping; if a state isn't mapped, it should be handled by another mapper
        """
        Mapping = {}
        for (i,state_comp) in enumerate(states[:,self.dim]):
            if (i in ignore) or not (self.low <= state_comp <= self.high):
                continue
            Mapping[i] = NodeDist(self.sink_node,1.0)
        return Mapping
        
    def nodes_to_states(self,nodes):
        """
        Does not encode a mapping for any nodes to states; all nodes are 
        """
        return {}
        
    def get_node_ids(self):
        return [self.sink_node]
        
    def get_num_nodes(self):
        return 1
        
    def get_node_states(self):
        return None        
        
class PiecewiseConstRegularGridNodeMapper(BasicNodeMapper):
    def __init__(self,*vargs):
        """
        This is a class that utilizes a REGULAR N-dimensional grid, and assigns everything with a cell to the same point.
        This representative node is at the center of mass of the cell.
        
        This is equivalent to a nearest-neighbor interpolation given the center of mass points.
        
        vargin be a list of triples: (low,high,N)
        This gives the range of discretization [low,high] and the number of cells in that range. There is a little
        fuzzing on the upper boundary so that it's not [low,high)
        
        NB: not that there are N cells; if one used np.linspace(low,high,N) as cut points, there would be N-1 cells
        """
        
        assert(len(vargs) >= 1)
        self.grid_desc = vargs
        self.grid_n = np.array([x[2] for x in vargs])        
        self.num_nodes = np.prod(self.grid_n)    
        
    def states_to_node_dists(self,states,**kwargs):
        """
        Assumes states to be an N x d np.array
        This is a partial mapping; if a state isn't mapped, it should be handled by another mapper
        """
        
        ignore = kwargs.get('ignore',set())
        
        # Discretize dimensions into grid coordinates
        (N,D) = states.shape
        GridCoord = np.empty(states.shape)
        for (d,(low,high,n)) in enumerate(self.grid_desc):
            GridCoord[:,d] = np.floor(((states[:,d] - low) * n) / (high - low))
            
            # Fudge to get the top of the final cell...
            up_mask = np.logical_and(high <= GridCoord[:,d], GridCoord[:,d] <= high + 1e-12)
            GridCoord[up_mask,d] = n-1 
                    
        coef = grid_coef(self.grid_n)        
        Nodes = GridCoord.dot(coef)  
        
        # Check if in bound
        InBounds = np.ones(N)
        for d in xrange(D):
            np.logical_and(InBounds, np.all(0 <= GridCoord,axis=1),InBounds)
            np.logical_and(InBounds, np.all(GridCoord < self.grid_n,axis=1),InBounds)
        
        # Create the mapping
        Mapping = {}
        for i in xrange(N):
            if i in ignore:
                continue
            if not InBounds[i]:
                # Should be handled before getting to the gridder
                continue
            Mapping[i] = NodeDist(int(Nodes[i]),1.0)
        return Mapping
        
    def states_to_transition_matrix(self,states,**kwargs):
        (N,D) = states.shape
        M = self.get_num_nodes()
        ND = self.states_to_node_dists(states,**kwargs)
        
        return build_interp_matrix_from_node_dists(N,M,D,ND)   
            
    def nodes_to_states(self,nodes):
        """
        Map nodes to point at the center of the cell
        """
        N = len(nodes)
        D = len(self.grid_desc)
        GridCoord = np.empty((N,D))        
        tmp = np.array(nodes)  # Copy nodes
        
        # Figure out the prefix array
        coef = grid_coef(self.grid_n)
            
        # Divide and mode to get the grid coords
        for d in xrange(D):
            GridCoord[:,d] = np.floor(tmp / coef[d])
            np.mod(tmp,coef[d],tmp)
                        
        # Convert from GridCoord to states
        for (d,(low,high,n)) in enumerate(self.grid_desc):
            GridCoord[:,d] *= float(high - low) / float(n) # Get the range right
            GridCoord[:,d] += low + float(high - low) / float(2*n) 
            # Add bottom of the range, plus half the grid's "delta"
            
        # Check if any of the points are out of bounds (NaN over the nonsense)
        OutOfBounds = np.logical_or(nodes < 0, nodes >= self.num_nodes)
        GridCoord[OutOfBounds,:] = np.NaN
            
        return GridCoord
        
    def get_node_ids(self):
        return xrange(self.num_nodes)
        
    def get_num_nodes(self):
        return self.num_nodes
        
    def get_node_states(self):
        # Build the D different uniformly spaced ranges
        linspaces = [np.linspace(low,high,n+1)[:-1] for (low,high,n) in self.grid_desc]
        # Turn these into a mesh
        meshes = np.meshgrid(*linspaces,indexing='ij')
        # Flatten each into a vector; concat; transpose
        node_states = np.column_stack(map(lambda x: x.ravel(),meshes))  
        shift = [(high - low) / float(2.0 * (n)) for (low,high,n) in self.grid_desc]
        return node_states + shift
        
    def get_dimension(self):
        return len(self.grid_desc)
        
class InterpolatedGridNodeMapper(BasicNodeMapper):
    def __init__(self,*vargs):
        """
        This is a class that utilizes a non-regular N-dimensional grid. Interstitial
        points are mapped to a particular convex combination of cell vertices defined
        by the multilinear weights.
        Compare this to PiecewiseConstGridNodeMapper that maps points to the center of the cell,
        rather than to a distribution over vertices. Additionally, these two mappers will have
        different numbers of points because of the vertex / centroid distinction.
        
        vargin be a list of sorted cut points: (c_1,...,c_k) for each dimension
        NB: this is a "fundemental" discretization, and we assume that node numbering starts from zero
        """
        assert(len(vargs) >= 1)
        self.grid_desc = vargs # Grid descriptions
        self.num_nodes = np.prod(map(len,self.grid_desc))
        
        self.node_states_cache = None
        self.node_id_cache = None
        
        self.__build_node_id_cache() # Cache mapping grid coords to node ids
        self.__build_node_state_cache() # Caches the states associated with each node
        
    def __str__(self):
        S = ['InterpolatedGridNodeMapper']
        S.extend(['({0}-{1:.2f})'.format(gd[0],gd[-1]) for gd in self.grid_desc])
        return ' '.join(S)
            
    def __grid_index_to_node_id(self,gid):       
        """
        Flattens an n-tuple representing the grid coordinates to a node-id
        """
        
        for (i,g_comp) in enumerate(gid):
            assert(0<=g_comp<len(self.grid_desc[i]))
        return self.node_id_cache[tuple(gid)]

            
    def __build_node_id_cache(self):
        """
        Builds a cache of how grid coords like (2,1,0,...) map to the node id
        
        The cache is a D-dimensional array, so conversion is just a table lookup
        Does this in fortran order
        """
        lengths = map(len, self.grid_desc)
        node_iter = xrange(self.num_nodes)
        self.node_id_cache = np.reshape(np.array(node_iter,dtype=np.uint32),lengths)
        
    def __build_node_state_cache(self):
        """
        Builds a cache of the states associated with each node.
        
        Cache is an N x D matrix, so lookup is just 
        """
        # Turn grids into meshes
        meshes = np.meshgrid(*self.grid_desc,indexing='ij')
        
        # Flatten each into a vector;
        self.node_states_cache = np.column_stack([mesh.ravel() for mesh in meshes])      
        
    def states_to_node_dists(self,states,**kwargs):    
        """
        Maps arbitrary states in the interior of the grid to a distribution of nodes
        defined by the multi-linear weights
        """
        ignore = kwargs.get('ignore',set())
        (N,D) = states.shape
        
        Mapping = {}
        # Iterate through each elements in the states matrix. Seems slow... vectorize somehow?
        for state_id in xrange(N):
            if state_id in ignore:
                continue      
                
            # Get the two cutpoints that sandwich the state in every dimension
            # E.g. greatest cutpoint smaller and least cutpoint larger
            Sandwiches = [] # SANDWEDGES
            for d in xrange(D):
                try:
                    (index,w_lo,w_hi) = one_dim_interp(states[state_id,d],self.grid_desc[d])
                except Error as e:
                    print e.strerror
                    raise AssertionError('Bad state {0}, (state={1},bad dim={2},grid={3})'\
                        .format(state_id,states[state_id,:],d,self.grid_desc[d]))
                Sandwiches.append((index,w_lo,w_hi))
            # Get the least node; e.g. in 2D:
            #   2 - 3
            #   |   |
            # ->0 - 1
            least_gid = map(lambda x: x[0],Sandwiches)
            dist = NodeDist()
            for gid_delta in itertools.product([0,1],repeat=D):
                curr_gid = map(sum,zip(least_gid,gid_delta)) # Add the delta to the least elem
                node_id = self.__grid_index_to_node_id(curr_gid) #Convert grid coord to node id
                dim_weights = map(lambda (i,x): Sandwiches[i][1+x], enumerate(gid_delta)) # Project out distances
                interp_weight = np.prod(dim_weights) # Multiply together
                if interp_weight > 0.0:
                    dist.add(node_id,interp_weight) # Add to the node distribution
            dist.normalize()
            dist.verify()
            Mapping[state_id] = dist
        return Mapping    
    
    def states_to_transition_matrix(self,states,**kwargs):
        (N,D) = states.shape
        M = self.get_num_nodes()
        ND = self.states_to_node_dists(states,**kwargs)
        
        return build_interp_matrix_from_node_dists(N,M,D,ND)        
                
    def nodes_to_states(self,nodes):
        """
        Map nodes to vertices
        
        Just uses the cache
        """        
        return self.node_states_cache[nodes,:]
        
    def get_node_ids(self):
        """
        Returns the range of nodes
        """
        return xrange(self.num_nodes)

    def get_num_nodes(self):
        return self.num_nodes
        
    def get_node_states(self):
        return self.node_states_cache
        
    def get_dimension(self):
        return len(self.grid_desc)
        
    def covers(self,state):
        for (d,gd) in enumerate(self.grid_desc):
            if not (gd[0] <= state[d] <= gd[-1]):
                return False
        return True
        
class InterpolatedRegularGridNodeMapper(InterpolatedGridNodeMapper):
    """
    Like InterpolatedGridNodeMapper, but assumes that the grid is regular
    """
    def __init__(self,*vargs):        
        assert(len(vargs) >= 1)
        self.grid_desc = vargs
        self.cell_n = np.array([x[2] for x in vargs]) 
        
        self.cut_point_n = [x + 1 for x in self.cell_n] # | x | x |
        self.cut_points = [np.linspace(low,high,n+1) for (low,high,n) in vargs]
        self.cell_width = [(high - low) / float(n)  for (low,high,n) in self.grid_desc]
        
        self.num_nodes = np.prod(self.cut_point_n)
        self.node_states_cache = np.empty(0)
        
    def __build_node_state_cache(self):
        """
        Builds a cache of the states associated with each node.
        
        Cache is an N x D matrix, so lookup is just 
        """
        # Turn grids into meshes
        meshes = np.meshgrid(*self.cut_points,indexing='ij')        
        
        # Flatten each into a vector;
        self.node_states_cache = np.column_stack([mesh.ravel() for mesh in meshes]) 

    def get_node_states(self):
        if not np.any(self.node_states_cache):
            self.__build_node_state_cache()
        return self.node_states_cache    
        
    def states_to_node_dists(self,states,**kwargs):
        if 1 == len(states.shape):
            states = states[np.newaxis,:]
        (N,d) = states.shape
        
        T = self.states_to_transition_matrix(states,**kwargs)
        
        mapping = {}
        for state_id in xrange(N):
            dist = NodeDist()
            # ...
        raise NotImplementedError() # This is probably dumb           
            
        
    def states_to_transition_matrix(self,states,**kwargs):
        """
        The assumption here is that either a state is in 'ignored' or it's in bounds.
        """
        ignore = kwargs.get('ignore',set())
        (N,D) = states.shape
        mask = np.ones(N,dtype=bool)
        for sid in ignore:
            mask[sid] = 0
            
        Lows = np.array([low for (low,high,n) in self.grid_desc])
        Highs = np.array([high for (low,high,n) in self.grid_desc])
        
        
        assert(np.all(states[mask,:] >= Lows - 1e-12))
        assert(np.all(states[mask,:] <= Highs + 1e-12))
        
        # This is least smallest grid coord in the dist            
        LeastGridCoords = np.empty((N,D)) # Lowest index
        W = np.empty((N,D,2)) # Weights
        for d in xrange(D):
            (low,high,n) = self.grid_desc[d]
            LeastGridCoords[mask,d] = np.floor((states[mask,d] - low) / self.cell_width[d] )
            LeastStateComp = LeastGridCoords[mask,d] * self.cell_width[d] + low
            assert(np.all(LeastGridCoords[mask,d] >=0 ))
            assert(np.all(states[mask,d] >= LeastStateComp - 1e-12))            
            
            diff = (states[mask,d] - LeastStateComp) / self.cell_width[d]
            W[mask,d,0] = 1.0 - diff
            W[mask,d,1] = diff            
            #Top
            top_mask = np.logical_and(states[mask,d] >= high-1e-12, states[mask,d] <= high + 1e-12)
            LeastGridCoords[top_mask,d] = n
            W[top_mask,d,0] = 1.0
            W[top_mask,d,1] = 0.0
            
        M = self.get_num_nodes()
        return build_interp_matrix_from_lgc_and_w(N,M,D,LeastGridCoords,W,self.cut_point_n,mask)
        
    def states_to_node_dists(self,states,**kwargs):
        raise NotImplementedError()
        
#########################################
# Some helper routines for node mapping #
#########################################        
        
def build_interp_matrix_from_node_dists(N,M,D,NodeDistDict):
    """
    Takes a dict of NodeDists and converts 
    """
    assert(N == len(NodeDistDict))
    T = scipy.sparse.lil_matrix((M,N))
    for (sid,nd)in NodeDistDict.items():
        for (nid,w) in nd.items():
            T[nid,sid] = w
    return T
        
def build_interp_matrix_from_lgc_and_w(NumStates,NumNodes,NumDims,LeastGridCoords,Weights,Lens,mask):
    """
    Build an MxN transition matrix from LeastGridCoord matrix and Weight tensor
    """
    assert((NumStates,NumDims) == LeastGridCoords.shape) # 
    assert((NumStates,NumDims,2) == Weights.shape) # Weight tensor
    assert((NumStates,) == mask.shape)
    
    V = mask.sum()
    valid_state_indices = np.arange(NumStates)[mask]
    
    assert((V,) == valid_state_indices.shape)
    Pow = 2**NumDims
    
    IJ = np.empty((2,0)) # Array indices
    Data = np.empty(0) # Data at those locations
    
    for (batch,delta) in enumerate(itertools.product([0,1],repeat=NumDims)):
        # Add the delta to the least grid coordinate
        GridCoords = LeastGridCoords[mask,:] + np.array(delta)
        assert((V,NumDims) == GridCoords.shape)
        
        # Get all the weights for these values
        # TODO: minor optimization; don't calculate for out-of-bound grid coords
        w = np.ones(V)
        for d in xrange(NumDims):
            w *= Weights[mask,d,delta[d]]
        
        # Some of these GridCoords will be out of the correct range
        # Mask these out (oob_mask assumes we've already filtered with mask)
        oob_mask = np.ones(V,dtype=bool)
        for d in xrange(NumDims):
            oob_mask[GridCoords[:,d] >= Lens[d]] = False
        
        # Convert the GridCoords to node ids
        node_ids = coords_to_id(GridCoords[oob_mask,:],Lens)
        assert((oob_mask.sum(),) == node_ids.shape)
        assert(np.amax(node_ids) < NumNodes)
        assert(np.amin(node_ids) >= 0)
        
        # Update the index and data structures
        ij = np.array([node_ids,valid_state_indices[oob_mask]])        
        IJ = np.hstack([IJ,ij])
        Data = np.hstack([Data,w[oob_mask]])
        
        # Check sizes
        assert(1 == len(Data.shape))
        assert((2,Data.size) == IJ.shape)
        
    # Create a csr matrix from these elements
    T = scipy.sparse.csr_matrix((Data,IJ),shape=(NumNodes,NumStates))
    return T
        
def grid_coef(Lens):
    """
    Builds weights for converting multi-dimensional grid coordinates into an id
    """
    coef = np.cumprod(np.flipud(Lens))
    coef = np.roll(coef,1) # circular shift
    coef[0] = 1.0
    return np.flipud(coef) # Reversed order; this is 'C' like order
  
def id_to_coords(node_id,Lens):
    """
    Expands a id into a C-ordered coordinate 
    (last dimension changes the most frequently)
    E.g. if the pair is (x,y) where x is row and y is column
    0 1 2 3 4
    5 6 7 ...    
    
    """
    N = len(Lens)
    coef = grid_coef(Lens)
    coords = np.empty(N)
    for i in reversed(xrange(N)):
        (coord,node_id) = divmod(node_id, coef[i])
        coords[i] = coord
    return coords        

def coords_to_id(coords,Lens):
    """
    Flattens a multi-dimensional coordinate into a C-ordered index
    """
    coef = grid_coef(Lens)    
    return coords.dot(coef)
    
def one_dim_interp(x,grid_desc):
    """
    Determines the cut points in grid_desc that float x falls between
    Returns a triple with the low index, and the distances from
    x to the two adjacent cut-points
    """
    assert(len(grid_desc) > 1) # Let's not be silly, here
    assert(grid_desc[0] <= x <= grid_desc[-1]) # Must be in bounds.

    index = bisect.bisect_left(grid_desc,x)-1
    
    # Boundary special cases
    if index == len(grid_desc)-1:
        assert(x == grid_desc[-1])
        index -= 1
    elif index == -1:
        assert(x == grid_desc[0])
        index = 0            
    assert(grid_desc[index] <= x<= grid_desc[index+1]) 
    
    
    gap = grid_desc[index+1] - grid_desc[index]
    ret = (index,(grid_desc[index+1] - x)/gap,(x - grid_desc[index])/gap)
    # Index in grid_desc, weights from vertices
    
    # If an endpoint ensure that things make sense
    if x == grid_desc[index]:
        assert(ret[1] == 1.0)
        assert(ret[2] == 0.0)
    if x == grid_desc[index+1]:
        assert(ret[1] == 0.0)
        assert(ret[2] == 1.0)
        
    assert(abs(sum(ret[1:]) - 1.0) < 1e-9)
        
    return ret