import numpy as np
import scipy.sparse as sps
import scipy as sp

import pickle
import matplotlib.pyplot as plt
import utils.processing

sol = np.load('data/test.sol.npy') # hard coded
data = pickle.load(open('data/test.pickle','rb'))

# Build policy from sol

mdp_obj = data['objects']['mdp']
A = mdp_obj.num_actions
n = mdp_obj.num_states
N = (A+1)*n

assert((N,) == sol.shape)

sol = np.reshape(sol,(n,(A+1)),order='F')
policy = np.argmax(sol[:,1::],axis=1)

# Build Markov chain matrix from policy and MDP

P = sps.lil_matrix((n,n))

# Uniform mixing.
for a in xrange(A):
    P += (1.0 / A) *mdp_obj.transitions[a]

# Utilize the given policy.
for i in xrange(n):
    P[:,i] = mdp_obj.transitions[policy[i]][:,i]

gamma = mdp_obj.discount
Et = sps.eye(n) - mdp_obj.discount * P

p = Et.dot(np.ones(n))

assert(not np.any(p <= 0))

p_frame = np.reshape(p[:441],(21,21),order='F')
#plt.pcolor(p_frame[1:-1,1:-1])
plt.pcolor(p_frame)
plt.colorbar()
plt.show()

np.save('data/p.npy',p)
