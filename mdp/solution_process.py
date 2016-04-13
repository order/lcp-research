import numpy as np
from mdp.state_functions import InterpolatedFunction

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


def build_functions(mdp_obj,
                    disc,
                    matrix):
    """
    Build interpolated functions from the columns of a matrix
    """
    (N,M) = matrix.shape
    assert(N == mdp_obj.num_states)
    flow_fns = []
    for a in xrange(M):
        fn = InterpolatedFunction(disc,matrix[:,a])
        flow_fns.append(fn)
    return flow_fns
