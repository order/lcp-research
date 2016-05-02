import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

import discrete
import utils.plotting
from utils.pickle import dump, load
from utils.plotting import cdf_points

def trajectory_plot(results):
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

def cdf_plot(returns):
    for (name,ret) in returns.items():
        (x,f) = cdf_points(ret)
        plt.plot(x,f,lw=2,alpha=0.1,label=name)[0]
    plt.legend(returns.keys(),loc='best')
    plt.show()

def percentile(returns):
    percents = [40,50,60]
    x = np.array(returns.keys())
    (N,) = x.shape
    D = np.array(returns.values())
    (n,m) = D.shape
    assert(N == n)
    
    P = np.percentile(D,percents,axis=1)
    for p in P:
        plt.plot(x,p,'o')
    plt.show()
    
root = 'data/hallway'
results = load(root + '.results.pickle')
returns = load(root + '.return.pickle')
states = load(root + '.starts.pickle')

percentile(returns)
