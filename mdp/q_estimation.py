import numpy as np

def get_q_vectors(mdp_obj,V):
    A = mdp_obj.num_actions
    N = mdp_obj.num_states
    assert((N,) == V.shape)
    
    c = mdp_obj.costs
    P = mdp_obj.transitions
    g = mdp_obj.discount

    Q = np.empty((N,A))
        
    for a in xrange(A):
        Q[:,a] = c[a] + g * (P[a].T).dot(V)

    return Q
