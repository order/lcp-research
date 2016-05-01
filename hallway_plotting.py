import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

import discrete
import utils.plotting
from utils.pickle import dump, load

root = 'data/hallway'
results = load(root + '.results.pickle')
returns = load(root + '.returns.pickle')
states = load(root + '.start_states.pickle')

N = len(results)

colors = iter(plt.cm.jet(np.linspace(0,1,N)))
handles = []
for (name,result) in results.items():
    data = result.states[:,0,:]
    color = next(colors)
    h = plt.plot(data.T,c=color,lw=2,alpha=0.1,label=name)[0]
    handles.append(h)
plt.legend(handles,results.keys(),loc='best')
plt.show()
