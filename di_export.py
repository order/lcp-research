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

MCTS_BUDGET = 1
NUM_STATES = 1
SIM_HORIZON = 100

#########################################
# Modes
ACTION_BEST = 1
ACTION_FREQ = 2
ACTION_ROLLOUT = 3

UPDATE_RET_V = 1
UPDATE_RET_Q = 2
UPDATE_RET_GAIN = 4

#########################################

class Params(object):
    def __init__(self,other=None):
        if other==None:
           self.default()
        else:
            self.copy(other)
        
    def default(self):
        self.p_scale = 1
        self.ucb_scale = 1
        self.rollout_horizon = 50
        
        self.init_q_mult = 0.1
        self.q_min_step = 0.0
        self.update_ret_mode = UPDATE_RET_GAIN
        
        self.action_select_mode = ACTION_ROLLOUT

    def copy(self,param):
        self.p_scale = param.p_scale
        self.ucb_scale = param.ucb_scale
        self.rollout_horizon = param.rollout_horizon
        
        self.init_q_mult = param.init_q_mult
        self.q_min_step = param.q_min_step
        self.update_ret_mode = param.update_ret_mode
        
        self.action_select_mode = param.action_select_mode 

    def to_list(self):
        L = []
        L.append(self.p_scale)
        L.append(self.ucb_scale)
        L.append(self.rollout_horizon)
        
        L.append(self.init_q_mult);
        L.append(self.q_min_step);
        L.append(self.update_ret_mode);

        L.append(self.action_select_mode)
        return L

    def to_array(self):
        L = self.to_list()
        return np.array(L)

    def perturb(self):
        if 0.5 > np.random.rand():
            self.p_scale = max(0,self.p_scale + 0.1*np.random.randn())
        if 0.5 > np.random.rand():
            self.ucb_scale = max(0,self.ucb_scale + 0.1*np.random.randn())
        if 0.5 > np.random.rand():
            if 0.5 > np.random.rand():
                self.rollout_horizon += 1
            else:
                self.rollout_horizon -= 1
            self.rollout_horizon = min(max(5,self.rollout_horizon),100)
        if 0.5 > np.random.rand():
            pert = self.init_q_mult + 0.1*np.random.randn()
            self.init_q_mult = max(0.0,min(1.0,pert))
        if 0.5 > np.random.rand():
            self.q_min_step = max(0,self.q_min_step + 0.1*np.random.randn())
        if 0.05 > np.random.rand():
            self.update_ret_mode = np.random.choice([UPDATE_RET_V,
                                                     UPDATE_RET_Q,
                                                     UPDATE_RET_GAIN])
        if 0.05 > np.random.rand():
            self.action_select_mode = np.random.choice([ACTION_BEST,
                                                        ACTION_FREQ])

    def __str__(self):
        return'\n'.join(['{0}:{1}'.format(k,v)
                         for (k,v) in self.__dict__.items()])
        

def marshal(static_params,starts,params,filename):
    marsh = Marshaller()
    # Grid
    marsh.extend(static_params);
    marsh.add(starts)
    marsh.extend(params.to_list())
    assert(23 == len(marsh.objects))
    marsh.save(filename)

def create_static_params():
    disc_n = 40 # Number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 5               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.01            # Control jitter 
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
    start_states = (np.random.rand(10,2) - 1)
    
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
    (p,d) = solve_with_kojima(mdp,1e-12,1000)

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
    static_params.append(q)
    static_params.append(flow)
    
    static_params.append(mcts_budget)
    static_params.append(sim_horizon)

    return static_params

if __name__ == "__main__":
    params = Params()
    static_params = create_static_params()

    #start_states = 2*np.random.rand(NUM_STATES,2)-1
    start_states = np.tile([0.4,0.1],(5,1)) 
    marshal(static_params,
            start_states,
            params,
            'cdiscrete/test.mcts')
