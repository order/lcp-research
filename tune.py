from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *

import glob
import multiprocessing
import os
import subprocess

import math
import matplotlib.pyplot as plt

MCTS_BUDGET = 500
WORKERS = multiprocessing.cpu_count()-1
BATCHES_PER_WORKER = 5
STATES_PER_BATCH = 3
TOTAL_ITER = 2
SIM_HORIZON = 100

root = os.path.expanduser('~/data/di') # root filename
driver = os.path.expanduser('~/repo/lcp-research/cdiscrete/driver')
#########################################
# Modes
UPDATE_RET_V = 1
UPDATE_RET_Q = 2
UPDATE_RET_GAIN = 4

ACTION_Q = 1
ACTION_FREQ = 2
#########################################

class Params(object):
    def __init__(self,other=None):
        if other==None:
           self.default()
        else:
            self.copy(other)
        
    def default(self):
        self.names = ['p_scale',
                      'ucb_scale',
                      'rollout_horizon',
                      'q_min_step',
                      'update_ret_mode',
                      'action_select_mode']
        self.x = np.array([5,
                           5,
                           25,
                           0.1,
                           UPDATE_RET_GAIN,
                           ACTION_Q])

    def copy(self,param):
        self.x = np.array(param.x)

    def to_array(self):
        return self.x

    def to_list(self):
        return [float(x) for x in self.x]

    def restart(self):
        self.x = np.array([np.random.uniform(0,50),
                           np.random.uniform(0,50),
                           np.random.randint(5,100),
                           np.random.uniform(0,0.25),
                           np.random.choice([UPDATE_RET_V,
                                             UPDATE_RET_Q,
                                             UPDATE_RET_GAIN]),
                           np.random.choice([ACTION_Q,
                                             ACTION_FREQ])])

    def perturb(self):
        self.x += np.array([0.2*np.random.randn(),
                            0.2*np.random.randn(),
                            np.random.choice([-1,0,1]),
                            0.01*np.random.randn(),
                            0,
                            0])
        self.x[-2] = np.random.choice([UPDATE_RET_V,
                                       UPDATE_RET_Q,
                                       UPDATE_RET_GAIN])
        self.x[-1] =  np.random.choice([ACTION_Q,
                                        ACTION_FREQ])                      
        self.x = np.maximum(0,self.x)

    def add_grad(self,grad):
        self.x -= grad
        self.x = np.maximum(0,self.x)
        

def marshal(static_params,starts,params,filename):
    marsh = Marshaller()
    # Grid
    marsh.extend(static_params);
    marsh.add(starts)
    marsh.extend(params.to_list())
    assert(21 == len(marsh.objects))
    marsh.save(filename)

def create_static_params():
    disc_n = 20 # Number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 5               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.1              # Control jitter 
    discount = 0.99           # Discount (\gamma)
    B = 5
    bounds = [[-B,B],[-B,B]]  # Square bounds, 
    cost_radius = 0.25        # Goal region radius
    
    actions = np.array([[-1],[0],[1]]) # Actions
    action_n = 3
    assert(actions.shape[0] == action_n)

    mcts_budget = MCTS_BUDGET

    # Uniform start states
    sim_horizon = SIM_HORIZON
    
    problem = make_di_problem(step_len,
                              n_steps,
                              damp,
                              jitter,
                              discount,
                              bounds,
                              cost_radius,
                              actions)
    # Generate MDP
    (mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)

    # Solve
    (p,d) = solve_with_kojima(mdp,1e-4,1000)

    # Build value function
    (v,flow) = split_solution(mdp,p)
    assert(np.all(flow > 0))
    q = q_vectors(mdp,v)
    assert(np.all(q > 0))

    static_params = []
    static_params.append(-B*np.ones(2,dtype=np.double)) # low
    static_params.append(B*np.ones(2,dtype=np.double)) # high
    static_params.append(disc_n*np.ones(2,dtype=np.double)) # num cells per dim

    # Physics params
    static_params.append(step_len)
    static_params.append(n_steps)
    static_params.append(damp)
    static_params.append(jitter)

    # Other MDP params
    static_params.append(cost_radius)
    static_params.append(discount)
    static_params.append(actions)

    # MCTS context
    static_params.append(v)
    static_params.append(flow)
    
    static_params.append(mcts_budget)
    static_params.append(sim_horizon)

    return static_params

def create_start_states(N,Batches):
    starts = []
    return [np.random.uniform(-5,5,size=(N,2)) for _ in xrange(Batches)]

def run_driver(filename):
    curproc = multiprocessing.current_process()
    devnull = open(os.devnull, 'w')
    cmd = [driver, filename]
    try:
        return subprocess.check_output(
            cmd, shell=False,
            stderr=devnull)
    except:
        quit()

def get_median_return(static_params,
                      start_states,
                      curr_params,
                      num_workers):
    # Init filenumber
    if not hasattr(get_median_return,'FILE_NUMBER'):
        get_median_return.FILE_NUMBER = 0
        
    # Write out config files
    files = []
    for start in start_states:
        filename = root + '/test.mcts.{0}'.format(
            get_median_return.FILE_NUMBER)
        get_median_return.FILE_NUMBER += 1
        marshal(static_params,
                start,
                curr_params,
                filename)
        files.append(filename)
    print 'Running {0} jobs on {1} workers'.format(len(start_states),
                                                   num_workers)

    # Simulate from config files
    pool = multiprocessing.Pool(num_workers)
    ret = pool.map(run_driver,files)
    pool.close()
    pool.join()

    returns = np.array([float(x) for x in ret])
    return np.median(returns);

def fake_median_return(static_params,
                      start_states,
                      curr_params,
                      num_workers):
    return np.linalg.norm(curr_params.to_array())

def get_gradient(static_params,
                 start_states,
                 ref_params,
                 ref_return,
                 num_workers,
                 num_samples):
    S = num_samples
    x = ref_params.to_array()
    (D,) = x.shape

    Y = np.empty((S,D))
    b = np.empty(S)
    pert_params = Params()
    for s in xrange(num_samples):
        pert_params.copy(ref_params)
        pert_params.perturb()

        y = pert_params.to_array()

        Y[s,:] = y - x
        b[s] = get_median_return(static_params,
                                 start_states,
                                 pert_params,
                                 num_workers) - ref_return
    ret = np.linalg.lstsq(Y,b)
    return ret[0]

def accept(last_return,curr_return):
    if last_return > curr_return:
        return True
    signed_error = (curr_return - last_return) / last_return
    perc = 0.1 * np.sqrt(signed_error)

    return perc > np.random.rand()
    
        

if __name__ == "__main__":
    static_params = create_static_params()
    
    total_iter = TOTAL_ITER
    num_workers = WORKERS
    batches = WORKERS * BATCHES_PER_WORKER
    points_per_batch = STATES_PER_BATCH
    
    best_params_mat = np.empty((total_iter,6))
    params_mat = np.empty((total_iter,6))
    best_return_vec = np.empty(total_iter)
    return_vec = np.empty(total_iter)

    start_states = create_start_states(points_per_batch,batches)

    best_params = Params()
    best_return = np.inf
    curr_params = Params()
    curr_return = get_median_return(static_params,
                                    start_states,
                                    curr_params,
                                    num_workers)
    for i in xrange(total_iter):
        print '-'*5,i,'-'*5
        last_params = Params(curr_params)
        last_return = curr_return

        # Occasionally restart to a random point
        if 0.001 > np.random.rand():
            print 'RESTART'
            curr_params.restart()
        elif 0.001 > np.random.rand():
            print 'RESET TO BEST'
            curr_params.copy(best_params)
        else:
            curr_params.perturb()

        curr_return = get_median_return(static_params,
                                        start_states,
                                        curr_params,
                                        num_workers)
        print 'Params:\n',curr_params.to_array()
        print 'Return:',curr_return
        if not accept(last_return,curr_return):
            curr_params.copy(last_params)
            curr_return = last_return

        # Always record best found so far.
        if curr_return <= best_return:
            best_params.copy(curr_params)
            best_return = curr_return
            np.save("best_found",best_params.to_array())
        best_params_mat[i,:] = best_params.to_array()
        best_return_vec[i] = best_return
        params_mat[i,:] = curr_params.to_array()
        return_vec[i] = curr_return

    print best_params.to_array()
    
    np.save('params',params_mat)
    np.save('return',return_vec)
    
    if False:
        fig = plt.figure()
        ax = fig.add_subplot('211')
        ax.plot(params_mat,alpha=0.25)
        ax.plot(best_params_mat,lw=2)
        ax = fig.add_subplot('212')    
        plt.plot(return_vec,alpha=0.25)
        plt.plot(best_return_vec,lw=2)
        plt.show()
