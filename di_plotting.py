import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

import discrete
import utils.plotting
from utils.pickle import dump, load

results = load('data/di.sim.pickle')
returns = load('data/di.returns.pickle')
states = load('data/di.start_states.pickle')

(N,D) = states.shape
assert(D == 2)

Grid = 500
for (name,result) in results.items():
    actions = result.actions
    (n,d,t) = actions.shape
    assert(N == n)
    assert(d == 1)
    F = actions[:,0,0] # First action

    cuts = [np.linspace(-2,2,Grid),
            np.linspace(-2,2,Grid)]
    X,Y = np.meshgrid(*cuts)
    P = discrete.make_points(cuts)
    
    K = 1 # Knumber of Knearest Kneighbors
    knn = neighbors.KNeighborsRegressor(K,weights='distance')
    Z = knn.fit(states,F).predict(P)
    Z = np.reshape(Z,(Grid,Grid),order='F')

    
    plt.pcolor(X,Y,Z)
    plt.scatter(states[:,0],
                states[:,1],
                c=F,s=10,lw=1)
    plt.title(name)

plt.show()

