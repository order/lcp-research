import numpy as np
import scipy.sparse as sps

class LCPBuilder(object):
    def __init__(self,mdp,disc,**kwargs):
        self.mdp = mdp
        self.val_reg = kwargs.get('val_reg',0.0)
        self.flow_reg = kwargs.get('flow_reg',1e-15)

        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions

        # List of nodes to be spliced from the model
        # Format: NODE_ID -> TERMINAL COST
        self.omitted_nodes = {}

    def build_uniform_state_weights(self):
        self.state_weights = np.ones(self.num_states)  
        
    def build_lcp(self):
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n

        # Build the LCP
        use_lil = True # COO method faster
        q = np.empty(N)
        q[0:n] = -self.state_weights

        row = []
        col = []
        data = []
        for a in xrange(A):
            shift = (a+1)*n
            E = self.get_E_matrix(a).tocoo()
            row.extend([E.row,E.col + shift])
            col.extend([E.col + shift,E.row])
            data.extend([E.data,-E.data])
            q[shift:(shift+n)] = self.costs[a]

        row.extend([np.arange(n),np.arange(n,N)])
        col.extend([np.arange(n),np.arange(n,N)])
        data.extend([val_reg*np.ones(n),
                     flow_reg*np.ones(A*n)])
        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        M = sps.coo_matrix((data,(row,col)),
                           shape=(N,N))

        name = 'LCP from {0} MDP'.format(self.name)
        return lcp.LCPObj(M,q,name=name)
    
    def add_drain(self,point,terminal_cost):
        # Get the ID associated with the point
        (D,) = point.shape
        ids = disc.to_indices(point[np.newaxis,:])
        assert((1,) == ids.shape)
        node_id = idxs[0]

        # Make sure that the id cleanly maps back to the
        # given point
        recovered_point = disc.indices_to_points(ids)
        assert((1,D) == recovered_point)
        assert(np.linalg.norm(point - recovered_point[0,:]) < 1e-12)

        self.omitted_nodes[node_id] = terminal_cost

    def remove_

        
