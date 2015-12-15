import numpy as np
from pickles import *
import matplotlib.pyplot as plt
import mdp.mdp as mdp

def basic_pickle_test():
    for _ in xrange(25):
        A = sps.csr_matrix(np.random.rand(25,26))
        x = csr_matrix_to_pickle_array(A)
        B = pickle_array_to_csr_matrix(x)
        
        D = (A - B).todense()
        rel_err = (np.linalg.norm(D) / np.linalg.norm(A.todense()))
        assert(rel_err < 1e-9)
  
def multi_pickle_test():
    for _ in xrange(25):
        A = []
        for i in xrange(5):
            A.append(sps.csr_matrix(np.random.rand(25,26)))
        
        x = multi_matrix_to_pickle_array(A)
        B = pickle_array_to_multi_matrix(x)
        assert(len(B) == len(A))
        for i in xrange(5):            
            D = (A[i] - B[i]).todense()
            rel_err = (np.linalg.norm(D) / np.linalg.norm(A[i].todense()))
            assert(rel_err < 1e-9) 

def mdp_test():
    T = sps.eye(5,format='csr')
    transitions = [T]*3
    costs = [np.ones(5)]*3
    actions = [-1,0,1]
    mdp_obj = mdp.MDP(transitions,costs,actions,name='moose')
    mdp_obj.write('foo.mdp.npz')
    mdp_obj2 = mdp.MDP('foo.mdp.npz')
    print mdp_obj2
    
basic_pickle_test()
multi_pickle_test()
mdp_test()
        