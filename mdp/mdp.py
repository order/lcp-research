import numpy as np
import scipy
import matplotlib.pyplot as plt
import lcp

class MDPValueIterSplitter(object):
    """
    Builds an LCP based on an MDP, and split it
    according to value-iteration based (B,C)
    """
    def __init__(self,MDP,**kwargs):
        self.MDP = MDP
        self.num_actions = self.MDP.num_actions
        self.num_states = self.MDP.num_states
        
        # Builds an LCP based on an MPD
        self.LCP = MDP.tolcp()
        self.value_iter_split()      
    
    def update(self,v):
        """
        Builds a q-vector based on current v
        """
        q_k = self.LCP.q + self.C.dot(v)
        return (self.B,q_k)    

    def value_iter_split(self):
        """
        Creates a simple LCP based on value-iteration splitting:
        M = [[0 I I], [-I 0 0], [-I 0 0]]
        """
        I_list = []
        P_list = []
        # Build the B matrix
        for i in xrange(self.num_actions):
            I_list.append(scipy.sparse.eye(self.num_states))
        self.B = mdp_skew_assembler(I_list)
        self.C = self.LCP.M - self.B

def mdp_skew_assembler(A_list):
    """
    Builds a skew-symmetric block matrix from a list of squares
    """
    A = len(A_list)
    (n,m) = A_list[0].shape
    assert(n == m) # Square
    N = (A+1)*n
    M = scipy.sparse.lil_matrix((N,N))
    for i in xrange(A):
        I = xrange(n)
        J = xrange((i+1)*n,(i+2)*n)
        M[np.ix_(I,J)] = A_list[i]
        M[np.ix_(J,I)] = -A_list[i]
    return M.tocsr()

class MDP(object):
    """
    MDP object
    """
    def __init__(self,transitions,costs,actions,**kwargs):
        self.discount = kwargs.get('discount',0.99)
        self.transitions = transitions
        self.costs = costs
        self.actions = actions
        self.name = kwargs.get('name','unnamed')
        
        A = len(actions)
        N = costs[0].size

        self.num_actions = A
        self.num_states = N

        assert(len(transitions) == A)
        assert(len(costs) == A)
        
        for i in xrange(A):
            assert(costs[i].size == N)
            assert(transitions[i].shape[0] == N)
            assert(transitions[i].shape[1] == N)            
        
    def get_action_matrix(self,a):
        """
        Build the action matrix E_a = I - \gamma * P_a^\top 
        """
        return scipy.sparse.eye(self.num_states) - self.discount * self.transitions[a].T

    def tolcp(self):
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n
        d = self.discount

        Top = scipy.sparse.coo_matrix((n,n))
        Bottom = None
        q = np.zeros(N)
        q[0:n] = -np.ones(n)
        for a in xrange(self.num_actions):
            E = self.get_action_matrix(a)
            
            # NewRow = [-E_a 0 ... 0]
            NewRow = scipy.sparse.hstack((-E,scipy.sparse.coo_matrix((n,A*n))))
            if Bottom == None:
                Bottom = NewRow
            else:
                Bottom = scipy.sparse.vstack((Bottom,NewRow))
            # Top = [...E_a^\top]
            Top = scipy.sparse.hstack((Top,E.T))
            q[((a+1)*n):((a+2)*n)] = self.costs[a]
        M = scipy.sparse.vstack((Top,Bottom))
        return lcp.LCP(M,q)

    def __str__(self):
        return '<{0} MDP with {1} actions and {1} states>'.\
            format(self.name, self.num_actions,self.num_states)
        
        

def value_iteration(MDP,**kwargs):
    """
    Do value iteration directly on the MDP
    """
    max_iter = kwargs.get('max_iter',int(1e3))
    abs_tol = kwargs.get('abs_tol',1e-6)

    v = kwargs.get('x0',np.zeros(MDP.num_points))

    gamma = MDP.discount
    P_T = []
    for a in xrange(MDP.num_actions):
        # Transpose, and convert to csr sparse
        P_T.append(scipy.sparse.csr_matrix(MDP.transitions[a].T))
    for i in xrange(max_iter):
        v_new = np.full(MDP.num_points,np.inf)
        for a in xrange(MDP.num_actions):
            v_new = np.minimum(v_new, MDP.costs[a] + gamma*P_T[a]*v)
        if np.linalg.norm(v_new - v) < abs_tol:
            return v_new
        v = v_new
    return v
            
            
def plot_value(G,v):
    """
    Plot a 2D value function based on a grid.
    """
    x_mesh,y_mesh = np.meshgrid(G.x_grid,G.y_grid)
    V = np.reshape(v[:-2],x_mesh.shape)
    #print V.shape
    #print x_mesh.shape
    plt.contour(x_mesh,y_mesh,V,25)
    plt.show()

def plot_trajectory(MDP, record):
    S = np.array(record.states)
    Diff = S - record.states[-1]
    n = MDP.num_states
    A = MDP.num_actions
    
    f, ax = plt.subplots()
    ax.semilogy(Diff[:n,:],'-b',linewidth=1,alpha=0.15)
    for a in xrange(A):
        ax.semilogy(Diff[(a+1)*n:(a+2)*n,:],'-r',alpha=0.15)
    ax.set_title('Difference between iterations and final value')
    plt.show()
    
    
    
