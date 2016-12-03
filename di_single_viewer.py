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

    if True:
        plt.figure()
        plt.suptitle("Primal")
        for a in xrange(A):
            plt.subplot(2,2,a+1)
            tmv.plot_vertices(nodes,faces,P[:,a])

    if False:
        plt.figure()
        plt.suptitle("Dual")
        for a in xrange(A):
            plt.subplot(2,2,a+1)
            tmv.plot_vertices(nodes,faces,D[:,a])  

    if True:
        plt.figure()
        plt.subplot(2,2,2)
        plt.title("Agg flow")
        tmv.plot_vertices(nodes,faces,np.sum(P[:,1:],axis=1))

        plt.subplot(2,2,3)
        plt.title("Policy")
        tmv.plot_faces(nodes,faces,sol.p_agg)

        plt.subplot(2,2,4)
        plt.title("Advantage function")
        tmv.plot_faces(nodes,faces,sol.adv)    

    if True:
        plt.figure()
        plt.suptitle("Heuristics")
        plt.subplot(2,2,1)
        plt.title("Min dual residual")
        R = np.minimum(D[:,1],D[:,2])
        tmv.plot_vertices(nodes,faces,R)
        idx = np.argmax(R)
        print "Maxmin dual residual:", idx, nodes[idx,:]
        plt.plot(nodes[idx,0],nodes[idx,1],'wo')

        plt.subplot(2,2,2)
        plt.title("Abs. Bellman residual")
        tmv.plot_vertices(nodes,faces,np.abs(sol.res))
        idx = np.argmax(np.abs(sol.res))
        print "Max Abs Bellman Residual:",idx, nodes[idx,:]
        plt.plot(nodes[idx,0],nodes[idx,1],'wo')

        plt.subplot(2,2,3)
        plt.title("Bellman residual")
        tmv.plot_vertices(nodes,faces,sol.res)
        idx = np.argmin(sol.res)
        print "Min Bellman Residual:",idx, nodes[idx,:]
        plt.plot(nodes[idx,0],nodes[idx,1],'ws')

        idx = np.argmax(sol.res)
        print "Max Bellman Residual:",idx, nodes[idx,:]
        plt.plot(nodes[idx,0],nodes[idx,1],'wo')

        plt.subplot(2,2,4)
        plt.title("Heuristic")
        H = np.abs(sol.res) * np.sqrt(np.sum(sol.P[:,1:],axis=1))
        tmv.plot_vertices(nodes,faces,H)
        
        idx = np.argmax(H)
        print "Max Heuristic:",idx, nodes[idx,:]
        plt.plot(nodes[idx,0],nodes[idx,1],'wo') 

    if False:
        assert N == sol.bases.shape[0]
        assert (A) == sol.bases.shape[2]
        K = sol.bases.shape[1]
        for i in xrange(1,K-1):
            plt.figure()
            plt.suptitle("Basis " + str(i))
            for a in xrange(A):
                if (i == K-1) and (a == 0):
                    continue
                plt.subplot(2,2,a+1)
                tmv.plot_vertices(nodes,faces,sol.bases[:,i,a])
    
    plt.show()
