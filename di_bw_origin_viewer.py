import sys
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from utils.archiver import *
from utils.plotting import cdf_points
import tri_mesh_viewer as tmv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    
    (_,file_base) = sys.argv

    sol = Unarchiver(file_base + ".data")

    coords = sol.coords
    l1 = sol.res_diff[:,0]
    l2 = sol.res_diff[:,1]
    linf = sol.res_diff[:,2]

    max_i = np.argmax(l1)
    min_i = np.argmin(l1)
    print "Min:", min_i,coords[min_i,:], l1[min_i]
    print "Max:", max_i,coords[max_i,:], l1[max_i]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:,0],coords[:,1],coords[:,2],s=75,c=l1,alpha=0.5)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    fig.colorbar(sc)
    plt.show()
