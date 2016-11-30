import sys
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from utils.archiver import *
from utils.plotting import cdf_points
import tri_mesh_viewer as tmv


def softmax(v,temp):
    w = np.exp(temp * np.abs(v))
    return w / np.sum(w)

def thresh(v,t):
    p = np.percentile(v,t*100)
    idx = np.where(v > p)[0]
    w = np.zeros(v.shape)
    w[idx] = v[idx]
    return w
    

if __name__ == "__main__":
    
    (_,file_base) = sys.argv

    sol = Unarchiver(file_base + ".data")
    (nodes,faces) = read_ctri(file_base + ".mesh")

    R = sol.residuals
    (N,I) = R.shape
    print sol.heuristics.shape
    if True:
        for i in xrange(0,I,2):
            plt.figure()
            plt.suptitle("Iter " + str(i))

            plt.subplot(2,2,1)
            plt.title("value")
            tmv.plot_vertices(nodes,faces, sol.primals[:,0,i])
        
            plt.subplot(2,2,2)
            plt.title("heuristic")
            H = sol.heuristics[:,i]
            tmv.plot_vertices(nodes,faces, thresh(H,0.9))

            plt.subplot(2,2,3)
            plt.title("policy")
            tmv.plot_vertices(nodes,faces,
                              np.argmax(sol.primals[:,1:,i],axis=1))
        
            plt.subplot(2,2,4)
            plt.title('residual')
            tmv.plot_vertices(nodes,faces,R[:,i])

    plt.figure()
    tmv.plot_vertices(nodes,faces, sol.primals[:,0,-1])
    C = sol.centers
    print C

    for i in xrange(len(C)):
        (a,b,c) = sol.params[i,:]
        s = 0.05
        plt.plot([C[i,0], C[i,0] + (s/b)*np.cos(a)],
                 [C[i,1], C[i,1] + (s/b)*np.sin(a)],'-k',lw=2)
        plt.plot([C[i,0], C[i,0] - (s/c)*np.sin(a)],
                 [C[i,1], C[i,1] + (s/c)*np.cos(a)],'-k',lw=2)
    plt.plot(C[:,0],C[:,1],'wo')

    plt.figure()
    tmv.plot_vertices(nodes,faces, sol.new_vec)


    plt.show()
