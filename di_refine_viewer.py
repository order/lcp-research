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

    for i in xrange(I):
        plt.figure()
        plt.title("Iter " + str(i))

        for a in xrange(3):
            plt.subplot(2,2,1+a)
            tmv.plot_vertices(nodes,faces, sol.primals[:,a,i])
        plt.subplot(2,2,4)
        tmv.plot_vertices(nodes,faces,R[:,i])

        

        
    plt.show()
