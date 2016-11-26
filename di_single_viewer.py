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
    plt.suptitle("Primal")
    for a in xrange(A):
        plt.subplot(2,2,a+1)
        tmv.plot_vertices(nodes,faces,P[:,a])

    plt.figure()
    plt.suptitle("Dual")
    for a in xrange(A):
        plt.subplot(2,2,a+1)
        tmv.plot_vertices(nodes,faces,D[:,a])  
        
    plt.figure()
    plt.subplot(2,2,1)
    plt.title("Min dual residual")
    R = np.minimum(D[:,1],D[:,2])
    tmv.plot_vertices(nodes,faces,R)

    plt.subplot(2,2,2)
    plt.title("Bellman residual")
    tmv.plot_vertices(nodes,faces,sol.res)

    plt.subplot(2,2,3)
    plt.title("Policy")
    tmv.plot_faces(nodes,faces,sol.p_agg)

    plt.subplot(2,2,4)
    plt.title("Advantage function")
    tmv.plot_faces(nodes,faces,sol.adv)

    plt.figure()
    plt.title("Heuristic")
    agg = np.sum(P[:,1:])
    H = np.maximum(np.abs(R),np.abs(sol.res)) * agg
    tmv.plot_vertices(nodes,faces,H)

    I = np.argmax(H)
    print "Heuristic max:"
    print "\tIndex:", I
    print "\tCoord:", nodes[I]
    
    plt.show()
