import numpy as np
import itertools
import operator
import bisect

class NodeMapper(object):
    """
    Abstract class defining state-to-node mappings. 
    For example, basic grid mappings or fixing OOB states.
    """
    def states_to_node_dists(self,states,**kwargs):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()
    def nodes_to_states(self,nodes):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()  
        
    def get_num_nodes(self):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()        
        
    def get_nodes(self):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()

    def get_node_states(self):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()
        
    def get_dimension(self):
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
            
    def __str__(self):
        return '{' + ', '.join(map(lambda x: '{0}:{1:.3f}'.format(x[0],x[1]), self.dist.items())) + '}'
        
class OOBSinkNodeMapper(NodeMapper):
    """
    Ensures that a state doesn't exceed an (axis-aligned) boundary by sending to sink state
    """
    def __init__(self,dim,low,high,sink_node):
        """
        low <= state[dim] <= high; otherwise map state to sink_node
        """
        self.dim = dim
        self.low = low 
        self.high = high
        self.sink_node = sink_node # this is the node id for the sink state
        
    def states_to_node_dists(self, states, ignore):
        """
        Assumes states to be an N x d np.array
        This is a partial mapping; if a state isn't mapped, it should be handled by another mapper
        """
        Mapping = {}
        for (i,state_comp) in enumerate(states[:,self.dim]):
            if (i in ignore) or (self.low <= state_comp <= self.high):
                continue
            Mapping[i] = NodeDist(self.sink_node,1.0)
        return Mapping
        
    def nodes_to_states(self,nodes):
        """
        Does not encode a mapping for any nodes to states; all nodes are 
        """
        return {}
        
    def get_nodes(self):
        return [self.sink_node]
        
    def get_num_nodes(self):
        return 1
        
    def get_node_states(self):
        return None        
        
class PiecewiseConstRegularGridNodeMapper(NodeMapper):
    def __init__(self,*vargs):
        """
        This is a class that utilizes a REGULAR N-dimensional grid, and assigns everything with a cell to the same point.
        This representative node is at the center of mass of the cell.
        
        This is equivalent to a nearest-neighbor interpolation given the center of mass points.
        
        vargin be a list of triples: (low,high,N)
        This gives the range of discretization [low,high) and the number of cells in that range
        """
        assert(len(vargs) >= 1)
        self.grid_desc = vargs
        
        self.num_nodes = np.prod(map(lambda x: x[2],vargs))
        
    def states_to_node_dists(self,states,**kwargs):
        """
        Assumes states to be an N x d np.array
        This is a partial mapping; if a state isn't mapped, it should be handled by another mapper
        """
        
        ignore = kwargs.get('ignore',set())
        
        # Discretize dimensions into grid coordinates
        (N,D) = states.shape
        GridCoord = np.zeros(states.shape)
        for (d,(low,high,n)) in enumerate(self.grid_desc):            
            GridCoord[:,d] = np.floor(((states[:,d] - low) * n) / (high - low))
            
        # Linearize multi-dimensional grid coordinates
        coef = np.ones(D)
        for d in xrange(D):
            coef[:d] *= self.grid_desc[d][2]
        Nodes = GridCoord.dot(coef)
        
        # Check if in bound
        InBounds = np.ones(N)
        for d in xrange(D):
            np.logical_and(InBounds, 0 <= GridCoord[:,d],InBounds)
            np.logical_and(InBounds, GridCoord[:,d] < self.grid_desc[d][2],InBounds)
        
        Mapping = {}
        for i in xrange(N):
            if i in ignore:
                continue
            if not InBounds[i]:
                # Raise warning here? Should be handled before getting to the gridder
                continue
                
            Mapping[i] = NodeDist(int(Nodes[i]),1.0)
        return Mapping
            
    def nodes_to_states(self,nodes):
        """
        Map nodes to point at the center of the cell
        """
        N = len(nodes)
        D = len(self.grid_desc)
        GridCoord = np.zeros((N,D))
        
        tmp = np.array(nodes)  # Copy nodes
        
        # Figure out the prefix array
        coef = np.ones(D)
        for d in xrange(D):
            coef[:d] *= self.grid_desc[d][2]
        print coef
            
        # Divide and mode to get the grid coords
        for (d,c) in enumerate(coef):
            GridCoord[:,d] = np.floor(tmp / c)
            np.mod(tmp,c,tmp)
                        
        # Convert from GridCoord to states
        for (d,(low,high,n)) in enumerate(self.grid_desc):
            GridCoord[:,d] *= float(high - low) / float(n) # Get the range right
            GridCoord[:,d] += low + float(high - low) / float(2*n) 
            # Add bottom of the range, plus half the grid's "delta"
            
        # Check if any of the points are out of bounds (NaN over the nonsense)
        OutOfBounds = np.logical_or(nodes < 0, nodes >= self.num_nodes)
        GridCoord[OutOfBounds,:] = np.NaN            
            
        return GridCoord
        
    def get_nodes(self):
        return xrange(self.num_nodes)
        
    def get_num_nodes(self):
        return self.num_nodes
        
    def get_node_states(self):
        # Build the D different uniformly spaced ranges
        linspaces = [np.linspace(*gd) for gd in self.grid_desc]
        # Turn these into a mesh
        meshes = np.meshgrid(*linspaces)
        # Flatten each into a vector; concat; transpose
        node_states = np.array(map(lambda x: x.flatten(),meshes)).T
        return node_states
        
    def get_dimension(self):
        return len(self.grid_desc)
        
class PiecewiseConstGridNodeMapper(NodeMapper):
        """
        This is a class that utilizes a non-regular N-dimensional grid. 
        It assigns everything with a cell to the same point.
        This representative node is at the center of mass of the cell.
        Compare this to PiecewiseConstRegularGridNodeMapper that assumes a special form for the grid
                
        vargin be a list of sorted cut points: (c_1,...,c_k) for each dimension
        """
        
class InterpolatedGridNodeMapper(NodeMapper):
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
        
        
    def __one_dim_interp(self,x,L):
        """
        Determines the cut points in L that float x falls between
        Returns a triple with the low index, and the distances from
        x to the two adjacent cut-points
        """
        
        assert(len(L) > 1) # Let's not be silly, here
        assert(x >= L[0]) # Must be in bounds.
        assert(x <= L[-1])
        
        index = bisect.bisect_left(L,x)-1
        if index == len(L)-1:
            # Right side of the last cell
            assert(x == L[-1])
            return (index-1,0.0,1.0)
        else:
            assert(x >= L[index])
            assert(x <= L[index+1])
            return (index,x - L[index], L[index+1] - x)
            
    def __grid_index_to_node_id(self,gid):       
        """
        Flattens an n-tuple representing the grid coordinates to a node-id
        """
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
        meshes = np.meshgrid(*self.grid_desc)
        # Flatten each into a vector; concat; transpose
        self.node_states_cache = np.array(map(lambda x: x.flatten(),meshes)).T        
        
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
                (index,w_lo,w_hi) = self.__one_dim_interp(states[state_id,d],self.grid_desc[d])
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
                
                interp_weight = reduce(operator.mul, dim_weights) # Multiply together
                dist.add(node_id,interp_weight) # Add to the node distribution
            dist.normalize() #
            dist.verify()
            Mapping[state_id] = dist
        return Mapping               
                
    def nodes_to_states(self,nodes):
        """
        Map nodes to vertices
        
        Just uses the cache
        """        
        return self.node_states_cache[nodes,:]
        
    def get_nodes(self):
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
  
def id_to_coords(node_id,Lens):
    """
    Expands a linear index into a fortran-ordered coordinate 
    (first dimension changes the most frequently)
    0 1 2 3
    4 5 6 7
    8 9 ...
    """
    coef = np.prod(Lens[:-1])
    coords = np.zeros(len(Lens))
    for i in reversed(xrange(len(Lens))):
        (coord,node_id) = divmod(node_id, coef)
        coef /= Lens[i]
        coords[i] = coord
    return coords        

def coords_to_id(coords,Lens):
    """
    Flattens a multi-dimensional coordinate into a fortran-ordered index
    """
    node_id = 0
    coef = 1
    for (i,coord) in enumerate(coords):
        node_id += coef * coord
        coef *= Lens[i]
    return node_id