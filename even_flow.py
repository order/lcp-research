import numpy as np
import pickle
import matplotlib.pyplot as plt
import utils.processing

sol = np.load('data/test.sol.npy') # hard coded
data = pickle.load(open('data/test.pickle','rb'))

# Build, from policy in sol and the MDP in the data
# The Markov chain diffusion matrix

mdp_obj = data['objects']['mdp']
A = mdp_obj.num_actions
n = mdp_obj.num_states
N = (A+1)*n
print A,n

assert((N,) == sol.shape)

sol = np.reshape(sol,(n,(A+1)),order='F')
policy = np.argmax(sol[:,1::],axis=1)
frame = np.reshape(policy[:441],(21,21))
plt.pcolor(frame.T)
plt.colorbar()
plt.show()
