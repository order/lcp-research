

##############################
# Iteration generators
    
def euler_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    step = kwargs['step']
    
    x = state.x
    w = M.dot(x) + q
    while True:
        x = nonneg_proj(x - step * w)
        w = M.dot(x) + q
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
def euler_linesearch_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    wolfe_const = kwargs.get('wolfe_const',1e-4)
    step_decay = kwargs.get('step_decay',0.9)
    min_step = kwargs.get('min_step', 1e-4)
    N = lcp_obj.dim
    
    x = state.x
    w = M.dot(x) + q
    while True:
        step = 1
        fx = fb_residual(x,w)
        grad_norm = np.linalg.norm(w)
        while True:
            x_cand = nonneg_proj(x - step * w)
            w_cand = M.dot(x_cand) + q
            fx_cand = fb_residual(x_cand,w_cand) # Min the Fischer-Burmeister residual
            if fx_cand <= fx - wolfe_const*step*grad_norm**2:
                break
            if step <= min_step:
                step = min_step
                break
            step *= step_decay
        w = w_cand
        x = x_cand
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state 
        
def euler_speedy(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    step_base = kwargs['step']
    
    Momentum = 0
    IncToken = 1
    DecelToken = 8
    BreakToken = 135
    IncThresh = 0.997
    BreakThresh = 0.97
    AngleEps = 7e-6
    ScoreEps = 5e-2
    Multiplier = 1.03
    
    x = state.x
    x_old = x
    w = M.dot(x) + q
    w = w / np.linalg.norm(w)
    w_old = w
    theta_old = 0
    
    score_best = float('inf');
    score_old = float('inf');
    while True:
        theta = w_old.dot(w)
        if theta > IncThresh and theta >= (1.0 - AngleEps)*theta_old:
            Momentum += IncToken
        if theta <= BreakThresh:
            Momentum -= BreakToken
            x = x_old
            w = M.dot(x) + q
            w = w / np.linalg.norm(w)
            
        step = step_base * Multiplier**Momentum
        x_old = x
        w_old = w
        theta_old = theta
        
        x = nonneg_proj(x - step * w)
        w = M.dot(x) + q
        w = w / np.linalg.norm(w)
        
        assert(isvector(x))
        assert(x.size == N)
        
        score = basic_residual(x,w)
        if score <= score_best:
            score_best = score
        if score > (1+ScoreEps)*score_old:
            Momentum -= DecelToken 
        print Momentum
            
        state.iter += 1
        state.x = x
        state.w = w
        yield state   
    
def projected_jacobi_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    assert(has_pos_diag(M))
    N = lcp_obj.dim
    D_inv = np.diag(np.diag(M))
    scale = kwargs.get('scale',1.0);    
    x = state.x
    w = M.dot(x) + q
    while True:
        x = nonneg_proj(x - scale*D_inv.dot(w))
        w = M.dot(x) + q

        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
def psor_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q
    assert(has_pos_diag(M))
    N = lcp_obj.dim
    relax = kwargs.get('omega',1.0)
    
    x = state.x
    while True:
        x = proj_forward_prop(x,M,q,relax)
        w = M.dot(x) + q
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
def extragrad_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q
    N = lcp_obj.dim
    step = kwargs['step']
    
    x = state.x
    w = M.dot(x) + q
    while True:
        y = nonneg_proj(x - step * w)
        v = M.dot(y) + q
        x = nonneg_proj(x - step * v)
        w = M.dot(x) + q
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
		
# Based on "Adaptive Restart for Accelerated Gradient Schemes" by O'Donoghue and Candes
# http://arxiv.org/pdf/1204.3982v1.pdf
def accelerated_prox_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    step = kwargs['step']
    restart = kwargs.get('restart',0.1)
    
    x = state.x
    w = M.dot(x) + q
    k = 1
    state.x_prev = x 
    y = x
    theta = 1
    
    while True:
        grad = M.dot(y) + q # grad f(y^k)        
        state.x_prev = x
        x = nonneg_proj(y - step * grad)
        theta_old = theta
        theta = quad(1,theta**2 - restart,-theta**2)[0]
        beta = theta_old * (1 - theta_old) / (theta_old**2 + theta)
        y = x + beta * (x - state.x_prev)
        
        if grad.dot(x - state.x_prev) > 0:
            theta = 1
            y = x
            print state.iter
        
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = grad
        yield state
