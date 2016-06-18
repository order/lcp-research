import numpy as np

def block_solution(mdp_obj,sol):
    """
    Reshape solution into matrix
    """
    A = mdp_obj.num_actions
    n = mdp_obj.num_states
    return sol.reshape(n,A+1,order='F')

def split_solution(mdp_obj,sol):
    """
    Break the solution into value and flow components
    """
    A = mdp_obj.num_actions
    n = mdp_obj.num_states
    N = (A+1)*n
    assert((N,) == sol.shape)
    V = sol[:n]
    F = sol[n:].reshape(n,A,order='F')
    assert(np.all(F[:,0] == sol[n:(2*n)]))
    return (V,F)

def q_vectors(mdp_obj,V):
    """
    Calculate Q vectors from V vector
    """
    A = mdp_obj.num_actions
    N = mdp_obj.num_states
    
    c = mdp_obj.costs
    P = mdp_obj.transitions
    g = mdp_obj.discount

    Q = np.empty((N,A))
        
    for a in xrange(A):
        Q[:,a] = c[a] + g * (P[a].T).dot(V)

    return Q

def reshape_physical(f,disc):
    (N,) = f.shape
    assert(N == disc.num_real_nodes())

    return f.reshape(*[n for n in disc.lengths])

def reshape_full(f,disc):
    (N,) = f.shape
    assert(N == disc.num_nodes())
    physical_f = f[:disc.num_real_nodes()]
    return reshape_physical(physical_f,disc)
    
