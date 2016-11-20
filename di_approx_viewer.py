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

    l1 = sol.data[:,0]
    l2 = sol.data[:,1]
    linf = sol.data[:,2]
     
    max_i = np.argmax(l1)
    min_i = np.argmin(l1)
    print "Min:", min_i, nodes[min_i,:],l1[min_i]
    print "Max:", max_i, nodes[max_i,:],l1[max_i]

    Interpolator = sp.interpolate.NearestNDInterpolator
    for (i,name) in enumerate(["l1","l2","linf"]):
        plt.subplot(2,2,i+1)
        
        interp = Interpolator(sol.centers,sol.data[:,i])
        Z = interp(nodes)
        plt.title(name)
        tmv.plot_vertices(nodes,faces,Z,cmap=plt.get_cmap('spectral'))
        
    plt.subplot(2,2,4)
    plt.title("Residual")
    tmv.plot_vertices(nodes,faces,sol.ref_res)

    plt.figure()
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
