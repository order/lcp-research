import lcp.gen as gen
import lcp.solvers as solvers
import lcp.util as util
import mdp

import functools
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pylab as plt
import time

import mdp.hillcar as hillcar
import mdp.hallway as hallway

class Nothing:
    pass
    
def compare_final(MDP,records):
    Img = []
    f, ax = plt.subplots() 
    for record in records:
        Img.append(record.states[-1])
    ax.imshow(Img)
    plt.show()
    
def value_iteration_spy():
    Actions = []
    N = 25
    A = 3
    for a in xrange(A):
        Actions.append(scipy.sparse.eye(N))
    M = mdp.mdp_skew_assembler(Actions)
    plt.spy(M)
    plt.show()
    
    
def run_MDP_value_compare():
    """ Compares MDP-split LCP iterations to value iteration w.r.t. v function
    """
    # Generate instance
    #N = 1500
    n = 15;

    #(MDP,G) = hillcar.generate_mdp(n,n)
    MDP = hallway.generate_mdp(n)
    LCP = MDP.tolcp()
    N = LCP.dim

    # Set up solver
    solver = solvers.iter_solver()
    solver.record_fns = [util.residual_recorder,util.state_recorder]
    solver.term_fns = [functools.partial(util.max_iter_term, 2000),\
        functools.partial(util.res_thresh_term, util.basic_residual,1e-6)]
   
    solver.params['centering_coeff'] = 0.1
    solver.params['linesearch_backoff'] = 0.8
    solver.params['mdp_split_inner_thresh'] = 1e-12
    solver.params['MDP'] = MDP


    solver_list = [solvers.mdp_value_iter,\
        solvers.mdp_ip_iter] 
    records = []
    for iter in solver_list:
        print 'Starting', iter.__name__
        start = time.time()
        solver.iter_fn = iter
        (record,state) = solver.solve(LCP,x0=np.ones(LCP.dim));
        elapsed = time.time() - start
        print 'Elapsed', elapsed
        records.append(record)
        
    linespec = ['-b','--r']
    X = np.array(records[0].states)
    Y = np.array(records[1].states)
    print X.shape
    print Y.shape
    assert(X.shape[1] == Y.shape[1])
    merge_iters = min(X.shape[0],Y.shape[0])
    
    """
    plt.plot(X[:merge_iters,:n],'-b',alpha=0.1)
    plt.plot(Y[:merge_iters,:n],'-r',alpha=0.1)
    plt.show()
    """    

    rel_diff = abs(X[:merge_iters,:n] - Y[:merge_iters,:n]) / Y[:merge_iters,:n]
    rel_diff_norm = np.linalg.norm(rel_diff,axis=1)
    plt.semilogy(rel_diff_norm)
    plt.show()
    
def run_compare():
    # Generate instance
    #N = 1500
    reg = 1e-9    
    n = 50;

    #(MDP,G) = hillcar.generate_mdp(n,n)
    MDP = hallway.generate_mdp(n)
    (M,q) = MDP.tolcp()
    N = M.shape[0]
    print 'Problem size',N
    #(M,q) = gen.rand_psd_lcp(N)
    #(M,q) = gen.rand_lpish(N,m)

    #M = M + reg*scipy.sparse.eye(*M.shape)
    #M = M + reg*np.eye(N)
    #L = util.max_eigen(M)
    #L = 0.5/reg


    # Set up solver
    solver = solvers.iter_solver()
    solver.record_fns = [util.residual_recorder,util.state_recorder]
    solver.term_fns = [functools.partial(util.max_iter_term, 2000),\
        functools.partial(util.res_thresh_term, util.basic_residual,1e-6)]


    #solver.params['step'] = 0.5/L

    # For linesearch
    #solver.params['min_step'] = 0.5/L

    # For PSOR
    #solver.params['omega'] = 1.4

    # For adaptive restart
    #solver.params['restart'] = 0.01
    
    solver.params['centering_coeff'] = 0.1
    solver.params['linesearch_backoff'] = 0.8
    solver.params['MDP'] = MDP


    solver_list = [solvers.basic_ip_iter,\
        solvers.kojima_ip_iter,\
        solvers.mdp_value_iter,\
        solvers.mdp_ip_iter] 
    records = []
    for iter in solver_list:
        print 'Starting', iter.__name__
        start = time.time()
        solver.iter_fn = iter
        (record,state) = solver.solve(M,q,x0=np.ones(q.size));
        elapsed = time.time() - start
        print 'Elapsed', elapsed
        records.append(record)
        
    for record in records:
        util.plot_state_img(record,max_len=27)
    #compare_final(MDP,records)
    
    

run_MDP_value_compare()
