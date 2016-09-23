import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.archiver import read_shewchuk

import sys
import argparse

from utils import make_points
from utils.archiver import Unarchiver
import tri_mesh_viewer as tmv

if __name__ == "__main__":
    filename = "/home/epz/data/minop/minop"
    (nodes,faces) = read_shewchuk(filename)
    plcp = Unarchiver(filename + '.plcp')
    #lcp = Unarchiver(filename + '.lcp')
    psol = Unarchiver(filename + '.psol')
    #sol = Unarchiver(filename + '.sol')

    V = nodes.shape[0]
    
    Phi = plcp.Phi.toarray()
    phiv = Phi[:V,:]
    phif = Phi[-V:,:]
    a = plcp.a
    b = plcp.b
    c = plcp.c
    v = np.minimum(b,c)

    #S = np.reshape(sol.p,(V,3),order='F')
    
    P = np.reshape(psol.p,(V,3),order='F')
    proj_v = phiv.dot(phiv.T.dot(P[:,0]))
    proj_f = phif.dot(phif.T.dot(P[:,1:]))
    plt.figure()
    for (i,f) in enumerate([v,
                            P[:,0],
                            v - P[:,0],
                            P[:,0]-proj_v]):
        plt.subplot(2,2,i+1)
        tmv.plot_vertices(nodes,faces,f)
    plt.suptitle('Value')
    plt.figure()
    for (i,f) in enumerate([P[:,1],
                            P[:,2],
                            P[:,1] - proj_f[:,0],
                            P[:,2] - proj_f[:,1]]):
        plt.subplot(2,2,i+1)
        tmv.plot_vertices(nodes,faces,f)
    plt.suptitle('Flow')
    plt.show()
                      
