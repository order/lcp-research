import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from lcp import *
from linalg import *
from utils import issorted

from sortedcontainers import SortedSet as sortedset

class LCPBuilder(object):
    def __init__(self,mdp,disc,**kwargs):
        self.mdp = mdp
        self.disc = disc
        self.val_reg = kwargs.get('val_reg',0.0)
        self.flow_reg = kwargs.get('flow_reg',1e-15)

        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions

        # List of nodes to be spliced from the model
        # Format: NODE_ID -> TERMINAL COST
        self.included_nodes = sortedset(xrange(self.num_states))
        self.omitted_nodes = {}

    def build_uniform_state_weights(self):
        scale = 1.0 / float(self.num_states)
        self.state_weights = scale * np.ones(self.num_states)
        
    def build(self):
        print 'Omitting', self.omitted_nodes.keys()
        idx = np.array(self.included_nodes,dtype=np.int)
        assert(issorted(idx))

        # Define matrix and block sizes
        block_size = idx.size # Use the included number 
        block_num = self.num_actions+1
        N = block_size*block_num # Total matrix size

        # First block of q is the state weight vector
        q = np.empty(N)
        q[:block_size] = -self.state_weights[idx]

        # Build the other blocks
        row = []
        col = []
        data = []
        for block in xrange(1,block_num):
            shift = block*block_size
            action = block - 1
            # Get the E block (I - g P)
            # Immediately slice off omitted nodes
            E = self.mdp.get_E_matrix(action)
            E = self.contract_sparse_square_matrix(E)

            # Add both E and -E.T to the growing COO structures
            row.extend([E.row,E.col + shift])
            col.extend([E.col + shift,E.row])
            data.extend([E.data,-E.data])

            # Add included costs to the q block
            q[shift:(shift+block_size)] = self.mdp.costs[action][idx]

        # Add regularization
        row.append(np.arange(N))
        col.append(np.arange(N))
        data.extend([self.val_reg*np.ones(block_size),
                     self.flow_reg*np.ones((block_num-1)*block_size)])

        # Concat COO structures into single vectors
        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        
        # Build the COO matrix
        M = sps.coo_matrix((data,(row,col)),
                           shape=(N,N),dtype=np.double)

        # Assemble into LCP object; return
        return LCPObj(M,q)

    ####################################################
    # For altering which nodes get included into the LCP

    def remove_node(self,node_id,terminal_cost):
        self.omitted_nodes[node_id] = terminal_cost
        self.included_nodes.remove(node_id)

        assert(len(self.omitted_nodes)
               == self.mdp.num_states - len(self.included_nodes))
    
    def add_drain(self,point,terminal_cost):
        # Get the ID associated with the point
        (D,) = point.shape
        ids = self.disc.to_indices(point[np.newaxis,:]).astype(np.int)
        assert((1,) == ids.shape)
        node_id = ids[0]

        # Make sure that the id cleanly maps back to the
        # given point
        recovered_point = self.disc.indices_to_points(ids)
        assert((1,D) == recovered_point.shape)
        assert(np.linalg.norm(point - recovered_point[0,:]) < 1e-12)

        self.remove_node(node_id,terminal_cost)

    def remove_oobs(self,terminal_cost):
        for node_id in self.disc.oob_range():
            self.remove_node(node_id,terminal_cost)

    def remove_unreachable(self,terminal_cost):
        T = self.mdp.aggregate_transitions()
        unreach = find_unreachable(T)
        assert(1 == len(unreach.shape))
        for idx in unreach:
            self.remove_node(idx,terminal_cost)

    def convert_variable(self,i):
        # Figure out what block and node variable corresponds to
        block_id = int(i / len(self.included_nodes))
        idx = i % len(self.included_nodes) # Mod out block
        node_id = self.included_nodes[idx]
        return (block_id, node_id)

    def contract_vector(self,f):
        """
        Remove nodes from single vector (e.g. value vector)
        """
        (n,) = f.shape
        assert(n == self.num_states)

        idx = np.array(self.included_nodes,dtype=np.int)
        return f[idx]
    
    def contract_block_vector(self,f):
        """
        Remove nodes from blocked vector (e.g. primal vector)
        """
        n = self.num_states
        A = self.num_actions
        N = n*(A+1)
        assert((N,) == f.shape)
        reshaped_f = np.reshape(f,(n,(A+1)),order='F')

        m = len(self.included_nodes)
        M = m * (A+1)
        idx = np.array(self.included_nodes,dtype=np.int)
        F = reshaped_f[idx,:]
        F = np.reshape(F,M,order='F')
        assert((M,) == F.shape)
        return F

    def contract_solution_block(self,sol):
        """
        Contract a n x (A+1) solution block
        """
        (n,a) = sol.shape
        
        assert(n == self.num_states)
        assert(a == self.num_actions + 1)
        
        idx = np.array(self.included_nodes,dtype=np.int)
        return sol[idx,:]
      
    
    def contract_block_matrix(self,B):
        """
        Use this for contracting solution N blocks
        (e.g. basis matrices)
        """
        (l,K) = B.shape
        
        n = self.num_states
        A = self.num_actions
        N = n*(A+1)
        assert(N == l)
        
        reshaped_B = np.reshape(B,(n,(A+1),K),order='F')

        m = len(self.included_nodes)
        M = m * (A+1)
        idx = np.array(self.included_nodes,dtype=np.int)
        
        B = reshaped_B[idx,...]
        B = np.reshape(B,(M,K),order='F')
        assert((M,K) == B.shape)
        return B

    def contract_sparse_square_matrix(self,M):
        (n,m) = M.shape
        assert(n == m)
        assert(n == self.num_states)

        idx = np.array(self.included_nodes,dtype=np.int)

        return spsubmat(M,idx,idx).tocoo()
    
    def expand_vector(self,f,pad_elem=np.nan):
        # Expand a vector corresponding to a single block
        (M,) = f.shape
        m = len(self.included_nodes)
        assert(M == m)

        N = self.num_states
        F = np.empty(N)

        idx = np.array(self.omitted_nodes.keys(),dtype=np.int)
        F[idx] = pad_elem
        idx = np.array(self.included_nodes)
        F[idx] = f

        return F
        
    def expand_block_vector(self, f, pad_elem=np.nan):
        # For padding out solutions from the LCP
        # so that it's the full size
        # I.e. map omitted nodes to to nan

        # Size checking
        assert(1 == len(f.shape))

        (M,) = f.shape
        m = len(self.included_nodes)
        A = self.mdp.num_actions
        assert(M == (A+1)*m)
        
        n = self.mdp.num_states
        N = (A+1)*n
        O = len(self.omitted_nodes)
        assert(M + (A+1)*O == N)

        # Convert block vector to matrix
        reshaped_f = np.reshape(f,(m,(A+1)),order='F')

        # Write information to blocks
        F = np.empty((n,A+1))
        idx = np.array(self.omitted_nodes.keys(),dtype=np.int)
        F[idx,:] = pad_elem
        idx = np.array(self.included_nodes,dtype=np.int)
        F[idx,:] = reshaped_f

        # Convert back to matrix
        F = np.reshape(F,(N,),order='F')
        return F

    def expand_block_matrix(self,F,pad_elem=np.nan):
        assert(2 == len(F.shape))

        (M,T) = F.shape
        m = len(self.included_nodes)
        A = self.mdp.num_actions
        assert(M == (A+1)*m)

        n = self.mdp.num_states
        N = (A+1)*n
        O = len(self.omitted_nodes)
        assert(M + (A+1)*O == N)

        # Convert block matrix to cube
        reshaped_F = np.reshape(F,(m,(A+1),T),order='F')

        F = np.empty((n,(A+1),T))
        idx = np.array(self.omitted_nodes.keys(),dtype=np.int)
        F[idx,...] = pad_elem
        idx = np.array(self.included_nodes,dtype=np.int)
        F[idx,...] = reshaped_F

        return np.reshape(F,(N,T),order='F')
         

def find_sinks(transition_matrix):
    # Find nodes that have no flow out
    T = transition_matrix
    (N,n) = T.shape
    assert(N == n)

    # Get the diagonal
    d = T.diagonal()
    
    # Find those super close to 1.0
    # I.e. only self-transition
    idx = np.argwhere(np.abs(d - 1.0) < 1e-15).squeeze()
    return idx

def find_unreachable(transition_matrix):
    # Find nodes that have no flow in
    T = transition_matrix
    (N,n) = T.shape
    assert(N == n)

    # Remove all diagonal entries
    d = T.diagonal()
    D = sps.diags(d,0)
    T -= D

    agg = np.array(T.sum(axis=1)).reshape(-1)
    idx = np.argwhere(np.abs(agg.reshape(-1)) < 1e-15).squeeze()
    print idx
    assert(1 == len(idx.shape))
    return idx    
    




