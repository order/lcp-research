import sys
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from utils.archiver import *
from utils.plotting import cdf_points
import tri_mesh_viewer as tmv

if __name__ == "__main__":
    
    (_,file_base) = sys.argv

    (nodes,faces) = read_ctri(file_base + ".mesh")
    print "Number of nodes:",len(nodes)
    print "Number of faces:",len(faces)
    sol = Unarchiver(file_base + ".data")

    P = sol.P
    D = sol.D
    
    (N,A) = P.shape
    plt.figure()
    plt.title("Primal")
    for a in xrange(A):
        plt.subplot(2,2,a+1)
        tmv.plot_vertices(nodes,faces,P[:,a])

    plt.figure()
    plt.title("Policy")
    policy = np.argmin(P[:,1:],axis=1)
    tmv.plot_vertices(nodes,faces,policy)

    plt.figure()
    plt.title("Dual")
    for a in xrange(A):
        plt.subplot(2,2,a+1)
        tmv.plot_vertices(nodes,faces,D[:,a])   
    
    plt.figure()
    plt.title("Residual")
    tmv.plot_vertices(nodes,faces,sol.res)

    
    plt.figure()
    plt.title("Min dual residual")
    R = np.minimum(D[:,1],D[:,2])
    tmv.plot_vertices(nodes,faces,R)

    plt.show()
