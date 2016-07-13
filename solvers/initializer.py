import numpy as np
import scipy.sparse as sps
from cvxopt import solvers,matrix,spmatrix

def generate_initial_feasible_points(M,q):
    """
    Generate initial strictly feasible points for the LCP (M,q)
    This is based on the linear program (P) in Kojima et al.'s
    "An interior point potential reduction algorithm for the
    linear complementarity problem" (p.5; End of section 3)
    """

    """
    max z
    s.t.
    y = Mx + q
    x,y >= z
    z <= 1

    x,y are (N,); z scalar
    2N+1 decision variables
    """

    (N,) = q.shape
    assert((N,N) == M.shape)

    c = np.zeros(2*N+1)
    c[-1] = -1 # Minimize -z <-> Maximize z

    #(x >= z) -> (0 >= z - x)
    Aineq = sps.bmat([[-sps.eye(N),None,np.ones((N,1))],
                      [None,-sps.eye(N),np.ones((N,1))],
                      [None,None,1]]).tocoo()
    assert((2*N+1,2*N+1) == Aineq.shape)
    # z <= 1
    bineq = -c
    

    # (y = Mx + q) -> (Mx - y = -q)
    Aeq = sps.bmat([[-M,sps.eye(N),np.zeros((N,1))]]).tocoo()
    beq = q
    assert((N,2*N+1) == Aeq.shape)

    solvers.options['feastol']=1e-12
    res = solvers.lp(matrix(c),
                     spmatrix(Aineq.data,
                              Aineq.row.tolist(),
                              Aineq.col.tolist(),
                              size=Aineq.shape),
                     matrix(bineq),
                     spmatrix(Aeq.data,
                              Aeq.row.tolist(),
                              Aeq.col.tolist(),
                              size=Aeq.shape),
                     matrix(beq))

    sol = np.array(res['x']).flatten()
    x = sol[:N]
    y = sol[N:-1]
    z = sol[-1]
    assert(y.shape == x.shape)
    assert((N,) == x.shape)

    print 'min(x)',np.min(x)
    print 'min(y)',np.min(y)
    print 'z',z
    print '||Mx+q-y||',np.linalg.norm(M.dot(x) + q -y)
    assert(np.all(x > 0))
    assert(np.all(y > 0))
    #assert(np.linalg.norm(M.dot(x) + q -y) < 1e-6)


    return (x,y,z)
