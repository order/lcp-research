##############################
# Basic infeasible-start interior point (p.159 / pdf 180 Wright)
def basic_ip_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    k = 0
    Top = sps.hstack([M,-sps.eye(M.shape[0])])
    x = np.ones(N)
    s = np.ones(N)
    sigma = kwargs.get('centering_coeff',0.01)
    beta = kwargs.get('linesearch_backoff',0.9)
    
    while True:
        assert(not any(x < 0))
        assert(not any(s < 0))
        
        mu = x.dot(s) / N # Duality measure
        
        # Construct matrix from invariant Top and diagonal bottom
        X = sps.diags(x,0)
        S = sps.diags(s,0)
        Bottom = sps.hstack([S, X])
        A = sps.vstack([Top,Bottom]).tocsr()
        r = s - M.dot(x) - q
        centering = -x*s + sigma*mu*np.ones(N)

        b = np.hstack([r, centering])
        
        # Solve, break solution into two parts
        d = sps.linalg.spsolve(A,b)        
        dx = d[:N]
        ds = d[N:]
        
        # Backtrack to maintain positivity.
        alpha = 1
        x_cand = x + alpha*dx
        s_cand = s + alpha*ds
        while any(x_cand < 0) or any (s_cand < 0):
            alpha *= beta
            x_cand = x + alpha*dx
            s_cand = s + alpha*ds
            
        x = x_cand
        s = s_cand
        state.x = x
        state.w = s
        state.iter = k
        k += 1
        yield state
