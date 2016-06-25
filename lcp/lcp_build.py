import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from lcp import *
from linalg import *
from utils import *
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
        idx = np.array(self.included_nodes)
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
            E = spsubmat(self.mdp.get_E_matrix(action),
                         idx,idx).tocoo()

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

    def convert_index(self,i):
        I = i % len(self.included_nodes) 
        return self.included_nodes[I]

    def expand(self, f, pad_elem=np.nan):
        # For padding out solutions from the LCP
        # so that it's the full size
        # I.e. map omitted nodes to to nan
        assert(1 == len(f.shape))

        (M,) = f.shape
        m = len(self.included_nodes)
        A = self.mdp.num_actions
        print 'expand sizes', M,m,A,(A+1)*m
        assert(M == (A+1)*m)
        
        n = self.mdp.num_states
        N = (A+1)*n
        O = len(self.omitted_nodes)
        assert(M + (A+1)*O == N)

        reshaped_f = np.reshape(f,(m,(A+1)),order='F')

        F = np.empty((n,A+1))
        idx = np.array(self.omitted_nodes.keys())
        F[idx,:] = pad_elem
        idx = np.array(self.included_nodes)
        F[idx,:] = reshaped_f

        F = np.reshape(F,(N,),order='F')
        return F      

def find_sinks(transition_matrix):
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
    




