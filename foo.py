import numpy as np
import matplotlib.pyplot as plt
from utils.archiver import *
from scipy.stats import pearsonr

import tri_mesh_viewer as tmv

def iqr(x,Y,marker):
    plt.gca()
    plt.plot(x,np.median(Y,0),'-' + marker,lw=2)
    plt.plot(x,np.percentile(Y,25,axis=0),'--' + marker,lw=2)
    plt.plot(x,np.percentile(Y,75,axis=0),'--' + marker,lw=2)


dir = '/home/epz/data/minop/'
files = [dir + 'flow_refine_0_false.exp_res',
         dir + 'flow_refine_10_false.exp_res',
         dir + 'flow_refine_15_false.exp_res']
colors = ['b','r','g']

plt.figure()
for (C,F) in zip(colors,files):
    unarch = Unarchiver(F)
    nb = unarch.num_basis[:-1]
    res = unarch.residuals
    (I,R) = res.shape

    for i in xrange(I):
        plt.plot(nb, res[i,:], C+'-',alpha=0.15)    
    iqr(nb,res,C)
plt.show()
