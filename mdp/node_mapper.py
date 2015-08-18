import numpy as np

class NodeMapper(object):
    """
    Abstract class defining state-to-node mappings. 
    For example, basic grid mappings or fixing OOB states.
    """
    def states_to_node_dists(self,states,ignore):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()
    def nodes_to_states(self,nodes):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()   
     
class NodeDist(object):
    """
    Stores information for a distribution over nodes
    """
    def __init__(self,*vargs,**kwargs):
        """
        Initializes NodeDist from either 
        """
        self.dist = {}
        self.add(*vargs,**kwargs)
        
    def add(self,*vargs,**kwargs):
        if len(vargs) > 1 or type(vargs[0]) not in [list,tuple]:
            raise TypeError('Input should be a single list of pairs',vargs)
            
        for (k,v) in vargs[0] + kwargs.items():
            if k in self.dist:
                raise KeyError('{0} already in dist table'.format(k))
            self.dist[k] = v
            
    def keys(self):
        return self.dist.keys()
            
    def verify(self):
        agg = 0.0
        for v in self.dist.values():
            agg += v
        assert(abs(agg - 1.0) < 1e-8)
        
    def __str__(self):
        return str(self.dist)
        
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
            Mapping[i] = NodeDist([(self.sink_node,1.0)])
        return Mapping
        
    def nodes_to_states(self,nodes):
        """
        Does not encode a mapping for any nodes to states; all nodes are 
        """
        return {}
        
class PiecewiseConstRegularGridNodeMapper(NodeMapper):
    def __init__(self,node_offset,*vargs):
        """
        This is a class that utilizes a regular N-dimensional grid, and assigns everything with a cell to the same point.
        This representative node is at the center of mass of the cell.
        
        node_offset is where the node range starts
        vargin be a list of triples: (low,high,N)
        This gives the range of discretization [low,high) and the number of bins N
        """
        assert(len(vargs) >= 1)
        self.grid_desc = vargs
        
        self.node_offset = node_offset
        self.num_nodes = reduce(lambda a,b: a*b, map(lambda x: x[2],vargs))
        self.node_max = node_offset + self.num_nodes
        
    def states_to_node_dists(self,states,ignore):
        """
        Assumes states to be an N x d np.array
        This is a partial mapping; if a state isn't mapped, it should be handled by another mapper
        """
        
        # Discretize dimensions into grid coordinates
        (N,D) = states.shape
        GridCoord = np.zeros(states.shape)
        for (d,(low,high,n)) in enumerate(self.grid_desc):            
            GridCoord[:,d] = np.floor(((states[:,d] - low) * n) / (high - low))
            
        # Linearize multi-dimensional grid coordinates
        coef = np.ones(D)
        for d in xrange(D):
            coef[:d] *= self.grid_desc[d][2]
        Nodes = GridCoord.dot(coef) + self.node_offset
        
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
                
            Mapping[i] = NodeDist([(int(Nodes[i]),1.0)])
        return Mapping
            
    def nodes_to_states(self,nodes):
        """
        Map nodes to point at the center of the cell
        """
        N = len(nodes)
        D = len(self.grid_desc)
        GridCoord = np.zeros((N,D))
        
        tmp = np.array(nodes) - self.node_offset # Copy nodes
        
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
        OutOfBounds = np.logical_or(nodes < self.node_offset, nodes > self.node_max)
        GridCoord[OutOfBounds,:] = np.NaN            
            
        return GridCoord
        
def nodemap_to_nodearray(node_map):
    """
    Simple utility for converting dict of deterministic node distributions into an array of node ids
    """
    Nodes = -np.ones(max(node_map.keys())+1)
    for (pos,n_dist) in sorted(node_map.items()):
        assert(len(n_dist.keys()) == 1) # Needs to be a deterministic mapping
        Nodes[pos] = n_dist.keys()[0]
    return Nodes
    