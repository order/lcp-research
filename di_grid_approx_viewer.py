import sys
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from utils.archiver import *
from utils.plotting import cdf_points
import tri_mesh_viewer as tmv

if __name__ == "__main__":
    
    (_,filename) = sys.argv

    sol = Unarchiver(filename)


    bw = sol.bandwidth
    B = bw.size
    grid_size = sol.grid_size
    G = grid_size.size
    
    data = sol.data
    assert (B,G,3) == data.shape

    for (i,name) in enumerate(['l1','l2','linf']):
        plt.figure()
        plt.title(name)
        for b in xrange(B):
            plt.semilogy(grid_size,data[b,:,i])
        plt.legend(bw,loc='best')
    plt.show()
            
