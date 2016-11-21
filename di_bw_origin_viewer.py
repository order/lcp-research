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

    bw = sol.bandwidths
    l1 = sol.data[:,0]
    l2 = sol.data[:,1]
    linf = sol.data[:,2]

    max_i = np.argmax(l1)
    min_i = np.argmin(l1)
    print "Min:", min_i,bw[min_i], l1[min_i]
    print "Max:", max_i,bw[max_i], l1[max_i]

    plt.figure()
    plt.suptitle('Additional basis at origin')
    for (i,name) in enumerate(['l1','l2','linf']):
        plt.subplot(2,2,i+1)
        plt.title(name)
        plt.semilogx(bw,sol.data[:,i],'.')
        plt.ylabel('Residual change')
        plt.xlabel('Bandwidth')    
    plt.show()
