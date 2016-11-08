import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.archiver import *
from utils.plotting import cdf_points
import tri_mesh_viewer as tmv

if __name__ == "__main__":
    
    (_,file_base) = sys.argv

    (nodes,faces) = read_ctri(file_base + ".mesh")
    sol = Unarchiver(file_base + ".sol")

    N = nodes.shape[0]
    assert 0 == sol.p.size % N
    A = sol.p.size / N

    P = np.reshape(sol.p,(N,A),order='F')
    D = np.reshape(sol.d,(N,A),order='F')

    Q = sol.Q
    assert((N,A) == Q.shape)

    if True:
        plt.figure()
        plt.suptitle('Projected Q')
        for a in range(A):
            plt.subplot(2,2,a+1)
            tmv.plot_vertices(nodes,faces,Q[:,a])
            print np.min(Q[:,a]), np.max(Q[:,a])
    
    for (X,name) in [(P,"Primal"),(D,"Dual")]:
        plt.figure()
        plt.suptitle(name)
        assert A == 3
        for a in xrange(A):
            plt.subplot(2,2,a+1)
            tmv.plot_vertices(nodes,faces,X[:,a])
            plt.title(str(a))
        if name == "Primal" and "ans" in sol.data:
            plt.subplot(2,2,4)
            tmv.plot_vertices(nodes,faces,sol.ans - X[:,0])
            plt.title("Residual")

        plt.figure()
        plt.title(name)
        for a in xrange(A):
            (x,f) = cdf_points(X[:,a])
            plt.semilogy(x,f)
        plt.legend(range(A),loc='best')
        
    
    plt.show()
