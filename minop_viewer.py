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
from utils.archiver import *
import tri_mesh_viewer as tmv

if __name__ == "__main__":
    (_,filename) = sys.argv
    (nodes,faces) = read_ctri(filename + ".mesh")
    sol = Unarchiver(filename + '.sol')

    V = nodes.shape[0]
    assert 0 == sol.p.size % V
    A = sol.p.size / V
    P = np.reshape(sol.p,(V,A),order='F')
    residual = sol.res
    assert V == residual.size

    plt.figure()
    plt.title("Residual l2-norm: " + str(np.linalg.norm(residual)))
    tmv.plot_vertices(nodes,faces,residual)

    plt.figure()
    plt.title("Agg flow")
    agg_flow = np.sum(P[:,1:],axis=1)
    tmv.plot_vertices(nodes,faces,agg_flow)
    plt.clim([0,np.max(agg_flow)])

    plt.figure()
    assert A <= 4
    for i in xrange(A):
        plt.subplot(2,2,(i+1))
        plt.title("Component " + str(i))
        tmv.plot_vertices(nodes,faces,P[:,i])

    plt.figure()
    plt.subplot(2,2,1)
    plt.title("Target function")
    tmv.plot_vertices(nodes,faces,sol.ans)
    plt.subplot(2,2,2)
    plt.title("Least-squares recon")
    tmv.plot_vertices(nodes,faces,sol.ls_recon)
    plt.subplot(2,2,3)
    plt.title("mPLCP answer")
    tmv.plot_vertices(nodes,faces,P[:,0])
    
    plt.show()
