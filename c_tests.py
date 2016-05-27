from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *
import cdiscrete as cd

import matplotlib.pyplot as plt

import time

# This file is for pushing V and flow vectors to C++ code for simulation

root = 'data/di/' # root filename
disc_n = 60
action_n = 3

# Generate problem
problem = make_di_problem()

# Generate MDP
(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)

# Solve
#(p,d) = solve_with_kojima(mdp,1e-12,1000)
#(v,F) = split_solution(mdp,p)
v = solve_with_value_iter(mdp,1e-6,1e4)

print 'Min:',min(v)
print 'Max:',max(v)

# Build value function
Q = q_vectors(mdp,v);

v_fn = InterpolatedFunction(disc,v);
q_fns = InterpolatedMultiFunction(disc,Q);

Ei = 100
cut = np.linspace(-3,3,Ei)
P = make_points([cut]*2);
Xi,Yi = np.meshgrid(cut,cut);
Z = np.argmin(q_fns.evaluate(P),axis=1);
Img = Z.reshape(Ei,Ei,order='F')
plt.pcolor(Xi,Yi,Img);
#plt.scatter(P[:,0], P[:,1], c=Z, lw=0)
plt.show()

quit()


(low,high,num) = zip(*disc.grid_desc)
low = np.array(low,dtype=np.double);
high = np.array(high,dtype=np.double);
num = np.array(num,dtype=np.uint64);
actions = mdp.actions.astype(np.double)

start_state = np.array([1.0,-np.sqrt(2)])

print 'V value:',v_fn.evaluate(start_state);
#for (i,q_fn) in enumerate(Q_fns):
#    print 'Action {0} Q value: {1}'.format(i,
#                                           q_fn.evaluate(start_state))

quit()

print 'Calling C++ function'
start = time.time()
cd.mcts_test(v,Q,F,actions,low,high,num,start_state)
print 'Elapsed time', time.time() - start
