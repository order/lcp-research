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
    for i in xrange(0,I,8):
        plt.figure()
        plt.suptitle("Iter " + str(i))
        
        plt.subplot(2,2,1)
        plt.title("value")
        tmv.plot_vertices(nodes,faces, sol.primals[:,0,i])

        plt.subplot(2,2,2)
        plt.title("agg flow")
        agg = np.sum(sol.primals[:,1:,i],1)
        tmv.plot_vertices(nodes,faces, np.log(agg))

        plt.subplot(2,2,3)
        plt.title("policy")
        tmv.plot_vertices(nodes,faces,
                          np.argmax(sol.primals[:,1:,i],axis=1))
        
        plt.subplot(2,2,4)
        plt.title('residual')
        tmv.plot_vertices(nodes,faces,R[:,i])


    plt.show()
