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
    
def run_compare():
    # Generate instance
    #N = 1500
    reg = 1e-9    
    n = 12;

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
        
    compare_final(MDP,records)
    

run_compare()