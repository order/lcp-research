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

    plt.figure()
    plt.suptitle("Reference solve")
    
    plt.subplot(2,3,1)
    plt.title("Value")
    tmv.plot_vertices(nodes,faces, sol.primal[:,0])
    
    plt.subplot(2,3,2)
    plt.title("Policy")
    tmv.plot_vertices(nodes,faces,
                      np.argmax(sol.primal[:,1:],axis=1))

    plt.subplot(2,3,3)
    plt.title("Min Flow Slack")
    tmv.plot_vertices(nodes,faces, np.min(sol.dual[:,1:],axis=1))
    
        
    plt.subplot(2,3,4)
    plt.title('Residual')
    tmv.plot_vertices(nodes,faces,sol.residual)

    plt.subplot(2,3,5)
    plt.title('Agg flow')
    tmv.plot_vertices(nodes,faces,np.sum(sol.primal[:,1:],axis=1))

        
    plt.subplot(2,3,6)
    plt.title('Advantage function')
    tmv.plot_vertices(nodes,faces,np.max(sol.dual[:,1:],axis=1)
                      - np.min(sol.dual[:,1:],axis=1))
    
    plt.show()
