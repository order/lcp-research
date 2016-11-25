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

    l1 = sol.data[:,:,0]
    l2 = sol.data[:,:,1]
    linf = sol.data[:,:,2]

    ############
    plt.figure()
    plt.suptitle("Residual change")
    for (i,name) in enumerate(["l1","l2","linf"]):
        plt.subplot(2,2,i+1)
        Z = np.min(sol.data[:,:,i],axis=1)
        plt.title(name)
        tmv.plot_vertices(nodes,faces,Z,cmap=plt.get_cmap('jet'))
    ############
    plt.figure()
    plt.suptitle("Minimizing bandwidth")
    for (i,name) in enumerate(["l1","l2","linf"]):
        plt.subplot(2,2,i+1)

        Z = sol.bandwidths[np.argmin(sol.data[:,:,i],axis=1)]
        Z = np.log(Z)
        plt.title(name)
        tmv.plot_vertices(nodes,faces,Z,cmap=plt.get_cmap('jet'))

    ############
    plt.figure()
    plt.suptitle('Residuals')
    plt.subplot(2,2,1)
    plt.title("Bellman Residual")
    tmv.plot_vertices(nodes,faces,sol.ref_res)
    
    plt.subplot(2,2,2)
    plt.title("Dual Min Residual")
    min_dual_res = np.min(sol.D[:,1:],axis=1)
    
    #min_dual_res = np.ones(len(nodes)) # Dummy
    tmv.plot_vertices(nodes,faces,min_dual_res)
    
    plt.subplot(2,2,3)
    plt.title("Diff")
    agg = np.sum(sol.P[:,1:],axis=1)
    H = (min_dual_res - sol.ref_res) * np.sqrt(agg)
    tmv.plot_vertices(nodes,faces,H)    

    ###########
    plt.figure()
    plt.suptitle("Reference problem information")
    plt.subplot(2,2,1)
    plt.title("Advantage")
    tmv.plot_faces(nodes,faces,sol.adv)

    plt.subplot(2,2,2)
    plt.title("Policy Agg")
    tmv.plot_faces(nodes,faces,sol.p_agg)

    plt.subplot(2,2,3)
    plt.title("Flow agg")
    tmv.plot_vertices(nodes,faces,np.sum(sol.P[:,1:],axis=1))

    plt.subplot(2,2,4)
    plt.title("Value function")
    tmv.plot_vertices(nodes,faces,sol.P[:,0])


    plt.show()
