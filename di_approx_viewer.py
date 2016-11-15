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
    sol = Unarchiver(file_base + ".data")

    l1 = sol.data[:,0]
    l2 = sol.data[:,1]
    linf = sol.data[:,2]

    Interpolator = sp.interpolate.NearestNDInterpolator
    for (i,name) in enumerate(["l1","l2","linf"]):
        plt.subplot(2,2,i+1)
        
        interp = Interpolator(sol.centers,sol.data[:,i])
        Z = interp(nodes)
        plt.title(name)
        tmv.plot_vertices(nodes,faces,Z)
        
    plt.subplot(2,2,4)
    plt.title("Residual")
    tmv.plot_vertices(nodes,faces,sol.ref_res)
    plt.show()
