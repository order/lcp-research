from mdp import *
from mdp.policies import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *

import matplotlib.pyplot as plt

import time
import math

root = 'data/hillcar/' # root filename

disc_n = 32
step_len = 0.05           # Step length
n_steps = 1               # Steps per iteration
jitter = 2              # Control jitter 
discount = 0.99            # Discount (\gamma)
bounds = [[-2,6],[-4,4]]  # Square bounds,
goal = np.array([0,0],dtype=np.double)
cost_radius = 0.1        # Goal region radius

action_n = 3
actions = np.linspace(-2,2,action_n).reshape(action_n,1)
assert(actions.shape[0] == action_n)

# Generate problem
problem = make_hillcar_problem(step_len,
                               n_steps,
                               jitter,
                               discount,
                               bounds,
                               goal,
                               cost_radius,
                               actions)

(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)
#v = solve_with_value_iter(mdp,1e-12,50000)
(p,d) = solve_with_kojima(mdp,1e-12,25000,1e-12,1e-12)
(v,flow) = split_solution(mdp,p)
v_fn = InterpolatedFunction(disc,v)
f_fns = InterpolatedMultiFunction(disc,flow)
q = q_vectors(mdp,v)
q_fns = InterpolatedMultiFunction(disc, q)
policy = IndexPolicyWrapper(MinFunPolicy(q_fns),
                           actions)
#policy = ConstantPolicy(np.array([0]))

N = 25
start_states = np.empty((N,2))
start_states[:,0] = np.random.uniform(-2,6,size=(N,))
start_states[:,1] = np.random.uniform(-4,4,size=(N,))

out = simulate(problem,
               policy,
               start_states,
               250)

G = 256
cuts = [np.linspace(bounds[0][0],bounds[0][1],G),
        np.linspace(bounds[1][0],bounds[1][1],G)]
P,(X,Y) = make_points(cuts,True)
VE = v_fn.evaluate(P)
VP = np.argmin(q_fns.evaluate(P),axis=1)
FE = f_fns.evaluate(P)
FP = np.argmax(FE,axis=1)
FS = np.sort(FE,axis=1)
AD = np.log(FS[:,-1] -FS[:,-2]+1e-22)
FA = np.log(np.sum(FE,axis=1) +1e-22)

v_img = np.reshape(VE,(G,G))
d_img = np.reshape(VP,(G,G))
f_img = np.reshape(FP,(G,G))
a_img = np.reshape(AD,(G,G))

images = [v_img,d_img,f_img,a_img]
cmaps = ['plasma','jet','jet','viridis']
titles = ['value','value policy','flow','log adv']
for (i,(img,cmap,title)) in enumerate(zip(images,cmaps,titles)):
    plt.subplot(1,len(images),i+1)
    plt.pcolormesh(X,Y,img,cmap=cmap)
    for i in xrange(N):
        plt.plot(out.states[i,0,:],
                 out.states[i,1,:],'w-',
                 alpha=0.15,lw=2)
    plt.title(title)
    plt.colorbar()

plt.show()
