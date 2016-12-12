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
    sol = Unarchiver(file_base + ".data")

    ref_P = sol.ref_P
    ref_D = sol.ref_D
    P = sol.P
    D = sol.D
    (N,A) = P.shape

    #######################################
    # Value
    plt.figure()
    plt.suptitle("Value information")
    plt.subplot(2,2,1)
    plt.title("Ref. value")
    tmv.plot_vertices(nodes,faces,ref_P[:,0])
    
    plt.subplot(2,2,2)
    plt.title("Approx. value")
    tmv.plot_vertices(nodes,faces,P[:,0])

    plt.subplot(2,2,3)
    plt.title("Difference")
    tmv.plot_vertices(nodes,faces,P[:,0] - ref_P[:,0])

    plt.subplot(2,2,4)
    plt.title("Bellman res")
    tmv.plot_vertices(nodes,faces,sol.res)

    #######################################
    # Flow
    plt.figure()
    plt.suptitle("Flow information")
    plt.subplot(2,2,1)
    plt.title("Ref. agg. flow")
    ref_agg = np.sum(ref_P[:,1:],axis=1)
    tmv.plot_vertices(nodes,faces,ref_agg)
    
    plt.subplot(2,2,2)
    plt.title("Approx. agg. flow")
    agg = np.sum(P[:,1:],axis=1)
    tmv.plot_vertices(nodes,faces,agg)

    plt.subplot(2,2,3)
    plt.title("Advantage")
    tmv.plot_faces(nodes,faces,sol.adv)

    ########################################
    # Dual
    plt.figure()
    plt.suptitle("Dual information")
    plt.subplot(2,2,1)
    plt.title("Min Dual")
    min_dual = np.min(D[:,1:],axis=1)
    tmv.plot_vertices(nodes,faces,min_dual)

    plt.suptitle("Dual policy")
    plt.subplot(2,2,2)
    plt.title("Min Dual")
    dual_policy = np.argmin(D[:,1:],axis=1)
    tmv.plot_vertices(nodes,faces,dual_policy)
    
    plt.subplot(2,2,3)
    plt.title("Policy")
    tmv.plot_faces(nodes,faces,sol.p_agg)

    plt.subplot(2,2,4)
    plt.title("Value residual")
    tmv.plot_vertices(nodes,faces,sol.D[:,0])

    ########################################
    # New vector
    plt.figure()
    plt.suptitle("New vectors")
    plt.subplot(1,2,1)
    tmv.plot_vertices(nodes,faces,sol.new_vects[:,0])
    plt.subplot(1,2,2)
    tmv.plot_vertices(nodes,faces,sol.new_vects[:,1]) 

    #######################################
    # Misc stuff
    plt.figure()
    plt.suptitle("Misc stuff")
    plt.subplot(2,2,1)
    plt.title("Old Bellman Res")
    tmv.plot_vertices(nodes,faces,sol.old_res)

    plt.subplot(2,2,2)
    plt.title("Bellman Res")
    tmv.plot_vertices(nodes,faces,sol.res)
    

    plt.suptitle("Res difference")
    plt.subplot(2,2,3)
    plt.title("Diff")
    tmv.plot_vertices(nodes,faces,sol.res - sol.old_res)
    
    plt.subplot(2,2,4)
    plt.title("Value res")
    tmv.plot_vertices(nodes,faces,sol.value_res)
    
    plt.show()
