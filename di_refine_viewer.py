import sys
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from utils.archiver import *
from utils.plotting import cdf_points
import tri_mesh_viewer as tmv

if __name__ == "__main__":
    
    (_,file_base) = sys.argv

    sol = Unarchiver(file_base + ".data")
    (nodes,faces) = read_ctri(file_base + ".mesh")

    R = sol.residuals
    (N,I) = R.shape
    for i in xrange(0,I):
        plt.figure()
        plt.suptitle("Iter " + str(i))
        
        plt.subplot(2,3,1)
        plt.title("Value")
        tmv.plot_vertices(nodes,faces, sol.primals[:,0,i])

        plt.subplot(2,3,2)
        plt.title("Policy")
        tmv.plot_vertices(nodes,faces,
                          sol.policies[:,i])

        plt.subplot(2,3,3)
        plt.title("Min Flow Slack")
        tmv.plot_vertices(nodes,faces, sol.min_duals[:,i])

        
        plt.subplot(2,3,4)
        plt.title('Residual')
        tmv.plot_vertices(nodes,faces,sol.residuals[:,i])

        plt.subplot(2,3,5)
        plt.title('Agg flow')
        tmv.plot_vertices(nodes,faces,np.sum(sol.primals[:,1:,i],axis=1))

        
        plt.subplot(2,3,6)
        plt.title('Advantage function')
        tmv.plot_vertices(nodes,faces,sol.advantages[:,i])
    plt.show()
