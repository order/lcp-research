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
    D = Unarchiver(file_base + ".data")

    (N,A) = D.basis.shape
    
    plt.figure()
    plt.suptitle("Value -> Flow");
    for a in xrange(A):
        plt.subplot(2,2,a+1);
        tmv.plot_vertices(nodes,faces,D.basis[:,a],
                          cmap=plt.get_cmap('jet'))
        
    plt.figure()
    plt.suptitle("Flow -> Value");
    for a in xrange(A):
        plt.subplot(2,2,a+1);
        tmv.plot_vertices(nodes,faces,D.ibasis[:,a],
                          cmap=plt.get_cmap('jet'))
    plt.show()
