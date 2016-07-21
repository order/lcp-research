import numpy as np
import scipy.sparse as sps
import cvxopt
from cvxopt import solvers,matrix,spmatrix
from lcp import LCPObj,ProjectiveLCPObj

def augment_lcp(lcp,scale,**kwargs):
    """
    Augment the LCP with an extra variable so that it is easier to initialize.

    Noting that Mx + (u - Mv - q) + q = y, we augment the LCP:

    0 <= [x]   |  [M (u - Mv - q)][x]  + [q] >= 0
    0 <= [x0] --- [0            s][x0] + [0] >= 0
    Then x = [v,1]' and y = [u,scale]'  are feasible initial
    points.

    So initializing with all variables at unity is a feasible
    start; and x0,y0 must go to zero.

    The scale term controls how imporant reducing x0y0 is, since
    x0y0 = scale * x0^2, which should be roughly equal to the central
    path term. It seems that setting scale >> 1 is advisable. 
    """
    
    M = lcp.M
    q = lcp.q
    (N,) = q.shape
    
    x = kwargs.get('x',np.ones(N))
    y = kwargs.get('y',np.ones(N))
    assert(np.all(x > 0))
    assert(np.all(y > 0))
    
    x0 = 1
    y0 = scale
    
    b = y - M.dot(x) - q
    assert((N,) == b.shape)
    assert((N,N) == M.shape)
    
    new_M = sps.bmat([[M,b[:,np.newaxis]],
                      [None,scale*np.ones((1,1))]])
    new_q = np.hstack([q,0])
    
    x = np.hstack([x,x0])
    y = np.hstack([y,y0])
    
    return LCPObj(new_M,new_q),x,y

def augment_plcp(plcp,scale,**kwargs):
    P = plcp.Phi
    U = plcp.U
    q = plcp.q
    (N,k) = P.shape
    assert((k,N) == U.shape)

    x = kwargs.get('x0',np.ones(N))
    y = kwargs.get('y0',np.ones(N))

    d = x - y
    proj_d = P.dot(P.T.dot(d))
    err = np.linalg.norm(d - proj_d)
    print 'Projection residual in x-y',err
    #assert(err < 1e-12)
    assert(np.all(x > 0))
    assert(np.all(y > 0))

    x0 = 1
    y0 = scale

    # b = y - Mx - q
    #   = y - (PUx + x - PPtx) - q
    #   = y -  PUx - x + PPtx - PPtq
    #   = PPt(y-x) + P(-Ux + Pt(x-q))
    # Assuming q and y-u in the span of P
    d = P.T.dot(y - x)
    b = d + -U.dot(x) + P.T.dot(x - q)
    assert((k,) == b.shape)
    
    # [P 0]
    # [0 1]
    new_P = sps.bmat([[P,None],
                      [None,np.ones((1,1))]])
    # [U b]
    # [0 s]
    new_U = sps.bmat([[U,b[:,np.newaxis]],
                      [None,scale*np.ones((1,1))]])
    new_PtPU = new_P.T.dot(new_P.dot(new_U))
    new_q = np.hstack([q,0])

    # Pw = u - u + q = q; w = Ptq
    # Pw0 = x0 - y0 = 1 - scale
    w = np.hstack([P.T.dot(q), 1.0 - scale])
    x = np.hstack([x,x0])
    y = np.hstack([y,y0])
    # w doesn't have to be +ve

    return ProjectiveLCPObj(new_P,new_U,new_PtPU,new_q),x,y,w

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

    solvers.options['max_iters'] = 100
    #solvers.options['kktsolver'] = 'robust'
    #solvers.options['abstol']=1e-9
    solvers.options['reltol']=1e-9
    solvers.options['feastol']=1e-9
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
                     matrix(beq),
                     solver='cvxopt')

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
