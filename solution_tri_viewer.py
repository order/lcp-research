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
    assert 0 == sol.p1.size % N
    A = sol.p1.size / N

    # RHS information
    if False:
        Q = sol.Q
        assert((N,A) == Q.shape)
    
        plt.figure()
        plt.suptitle('Projected Q')
        for a in range(A):
            plt.subplot(2,2,a+1)
            tmv.plot_vertices(nodes,faces,Q[:,a])

    # Before and after primal/dual info
    if True:        
        P1 = np.reshape(sol.p1,(N,A),order='F')
        D1 = np.reshape(sol.d1,(N,A),order='F')
        P2 = np.reshape(sol.p2,(N,A),order='F')
        D2 = np.reshape(sol.d2,(N,A),order='F')
        for (X,name) in [(P1,"Primal 1"),(D1,"Dual 1"),
                         (P2,"Primal 2"),(D2,"Dual 2")]:
            plt.figure()
            plt.suptitle(name)
            assert A == 3
            for a in xrange(A):
                plt.subplot(2,2,a+1)
                if a > 0:
                    assert np.all(X[:,a] > 0)
                    tmv.plot_vertices(nodes,faces,np.log(X[:,a]))
                else:
                    tmv.plot_vertices(nodes,faces,X[:,a])
                plt.title(str(a))

    # Reference plot (smoothed LCP)
    if False:
        R = np.reshape(sol.rp,(N,A),order='F')
        
        plt.figure()
        plt.suptitle('Reference primal')
        assert A == 3
        for a in xrange(A):
            plt.subplot(2,2,a+1)
            if a > 0:
                assert np.all(R[:,a] > 0)
                tmv.plot_vertices(nodes,faces,np.log(R[:,a]))
            else:
                tmv.plot_vertices(nodes,faces,R[:,a])
            plt.title(str(a))
                      

    # Residual information
    if True:
        plt.figure()
        plt.suptitle('Residual and heuristic')
        plt.subplot(2,2,1);
        plt.title("Bellmen residual 1")
        tmv.plot_vertices(nodes,faces,sol.res1)
        
        plt.subplot(2,2,2);
        plt.title("Bellmen residual 2")
        tmv.plot_vertices(nodes,faces,sol.res2)

        plt.subplot(2,2,3);
        plt.title("Bellmen residual difference")
        tmv.plot_vertices(nodes,faces,sol.res_diff)       

    if False:
            plt.figure()
            plt.title(name)
            for a in xrange(A):
                (x,f) = cdf_points(X[:,a])
                plt.semilogy(x,f)
            plt.legend(range(A),loc='best')
    
    plt.show()
