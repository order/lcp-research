import numpy as np
import scipy.sparse as sps
import scipy as sp

import pickle
import matplotlib.pyplot as plt
import utils.processing
import lcp
from solvers import *
from solvers.kojima import KojimaIPIterator

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
#for a in xrange(A):
#    P += (1.0 / A) *mdp_obj.transitions[a]

# Utilize the given policy.
for i in xrange(n):
    P[:,i] = mdp_obj.transitions[policy[i]][:,i]

# Form the E matrix
gamma = mdp_obj.discount
Et = sps.eye(n) - gamma * P

#Form M:
eta = 0.1
M = sps.bmat([[sps.eye(n), None, -Et.T],
              [None, eta*sps.eye(n), sps.eye(n)],
              [Et, -sps.eye(n), None]])
q = np.zeros(3*n)
q[:n] = -1

lcp_obj = lcp.LCPObj(M,q)
kojima_iter = KojimaIPIterator(lcp_obj)
solver = IterativeSolver(kojima_iter)
solver.termination_conditions.append(MaxIterTerminationCondition(200))
solver.termination_conditions.append(PrimalChangeTerminationCondition(1e-6))
solver.notifications.append(PrimalDiffAnnounce())
solver.solve()

x = kojima_iter.get_primal_vector()
f = x[:n]
p = x[n:(2*n)]

assert(not np.any(p <= 0))
assert(not np.any(f <= 0))

print 'p + dPf - f Residual:', np.linalg.norm(p + gamma * P.dot(f) - f)

#p_frame = np.reshape(p[:441],(21,21),order='F')
#plt.pcolor(p_frame[1:-1,1:-1])
#plt.pcolor(p_frame)
#plt.colorbar()
#plt.show()

print np.max(p)
print np.min(p)
np.save('data/p.npy',p)
