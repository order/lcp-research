import solver

def mdp_ip_iter(lcp_obj,state,**kwargs):
    """
    Splits an lcp_obj based on an MDP into (B,C) blocks that correspond to
    value iteration. Cottle 5.2 has more details on general splitting
    methods like PSOR
    """
    assert('MDP' in kwargs)
    MDP = kwargs['MDP'] # Must be present; don't want to do size inference

    M = lcp_obj.M
    q = lcp_obj.q
    sigma = kwargs.get('centering_coeff',0.1)
    beta = kwargs.get('linesearch_backoff',0.8)
    
    solve_thresh = kwargs.get('mdp_split_inner_thresh',1e-6)
    
    N = lcp_obj.dim
    x = np.ones(N)
    y = np.ones(N)
    
    split = mdp.MDPValueIterSplitter(MDP) # Splitter based on value iter

    Total_I = 0
    Outer_I = 0
    while True:
        # Update q split based on current x
        (B,q_k) = split.update(x) 
        # Use Kojima's UIP to solve lcp_obj(B,q_k)
        inner_solver = kojima_ip_iter(lcp.LCPObj(B,q_k),state,**kwargs)
        Inner_I = 0        
        
        # Want both complementarity and residual to be small before
        # stopping the inner iter
        while x.dot(y) >= solve_thresh\
            or np.linalg.norm(y - (q_k + B.dot(x))) > solve_thresh:
            state = inner_solver.next()
            x = state.x
            y = state.w
            Inner_I += 1
            Total_I += 1
        
        #Use the actual Mx+q rather than IP solver's 
        state.w = M.dot(x) + q
        state.iter = Outer_I # Could use other counters
        yield state  
        Outer_I += 1
