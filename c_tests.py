from mdp import *
from mdp.policies import BangBangPolicy
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *
import cdiscrete as cd

import matplotlib.pyplot as plt

import time
import math

# This file is for pushing V and flow vectors to C++ code for simulation

root = 'data/di/' # root filename
disc_n = 320
action_n = 3
discount = 0.99
tail_error = 0.05
M = 10000 # Controls number of samples.
horizon = int(math.log(tail_error*(1.0 - discount)) / math.log(discount))

# Generate problem
problem = make_di_problem(discount)

# Generate MDP
(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)

# Solve
#(p,d) = solve_with_kojima(mdp,1e-12,1000)
#(v,F) = split_solution(mdp,p)
v = solve_with_value_iter(mdp,1e-6,2500)

print 'Min:',min(v)
print 'Max:',max(v)

# Build value function
Q = q_vectors(mdp,v);

v_fn = InterpolatedFunction(disc,v);
q_fns = InterpolatedMultiFunction(disc,Q);

(low,high,num) = zip(*disc.grid_desc)
low = np.array(low,dtype=np.double);
high = np.array(high,dtype=np.double);
num = np.array(num,dtype=np.uint64);
actions = mdp.actions.astype(np.double)

q_policy =  IndexPolicyWrapper(MinFunPolicy(q_fns),
                               mdp.actions)
bang_policy = BangBangPolicy()

print "Horizon:", horizon
start_states = np.vstack([10*np.random.rand(M,2) - 5,
                          4*np.random.rand(M,2) - 2,
                          2*np.random.rand(M,2) - 1,
                          np.random.rand(M,2) - 0.5])

# Simulate
q_result = simulate(problem,
                  q_policy,
                  start_states,
                  horizon);
b_result = simulate(problem,
                    bang_policy,
                    start_states,
                    horizon)
q_ret = discounted_return(q_result.costs,problem.discount);
b_ret = discounted_return(b_result.costs,problem.discount);

# Estimate the return
est = v_fn.evaluate(start_states)

# KNN interp
G = 250
K = 7
(XI,YI,Z_est) = scatter_knn(est,start_states,K,G)
(XI,YI,Z_q_emp) = scatter_knn(q_ret,start_states,K,G)
(XI,YI,Z_b_emp) = scatter_knn(b_ret,start_states,K,G)

fig = plt.figure();
fig.suptitle('Discount={0},NumCells={1}'.format(discount,disc_n))
imgs = [Z_est,
        Z_q_emp,
        Z_b_emp,
        Z_est - Z_q_emp,
        Z_est - Z_b_emp,
        Z_q_emp - Z_b_emp]
titles = ['V estimate',
          'Q-policy empirical',
          '!!-policy empirical',
          'V - Q',
          'V - !!',
          'Q - !!']

AbsImg = np.vstack(imgs[:3])
DiffImg = np.vstack(imgs[3:])
abs_lo = np.min(AbsImg[:])
abs_hi = np.max(AbsImg[:])
diff_lo = np.min(DiffImg[:])
diff_hi = np.max(DiffImg[:])

for i in xrange(3):
    ax = fig.add_subplot('23' + str(i+1))
    cax=ax.pcolor(XI,YI,imgs[i],vmin=abs_lo,vmax=abs_hi)
    ax.set_title(titles[i])
    fig.colorbar(cax)
    
for i in xrange(3,6):
    ax = fig.add_subplot('23' + str(i+1))
    cax=ax.pcolor(XI,YI,imgs[i],cmap='inferno',vmin=diff_lo,vmax=diff_hi)
    ax.set_title(titles[i])
    fig.colorbar(cax)
    
plt.show()
plt.savefig('data/images/v_q_bang_compare_{0}.png'.format(disc_n))
