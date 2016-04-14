import numpy as np
import pickle
from argparse import ArgumentParser
from sklearn import neighbors

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import discrete

def qq_plot(V,R):
    assert((N,) == V.shape)
    assert((N,) == R.shape)
    l = min(np.min(V),np.min(R))
    u = max(np.max(V),np.max(R))
    plt.plot(V,R,'b.')
    plt.plot([l,u],[l,u],':r')
    plt.xlabel('Expected')
    plt.ylabel('Empirical')
    plt.title('Expected return vs empirical return')
    plt.show()



if __name__ == '__main__':
    parser = ArgumentParser(__file__,\
        'Form a policy from value function')
    parser.add_argument('sim_in_file',
                        metavar='FILE',
                        help='simulation data in file')
    parser.add_argument('problem_in_file',
                        metavar='FILE',
                        help='problem in file')
    parser.add_argument('img_out_file',
                        metavar='FILE',
                        help='image out file')
    parser.add_argument('-t','--title',
                        default='No title',
                        help='title')
    parser.add_argument('-x',
                        default='x',
                        help='x axis label')
    parser.add_argument('-y',
                        default='y',
                        help='y axis label') 
    args = parser.parse_args()

    FH = open(args.sim_in_file,"r")
    (V,R,states) = pickle.load(FH)
    FH.close()

    FH = open(args.problem_in_file,"r")
    problem = pickle.load(FH)
    FH.close()

    N = 250

    # Plotting actual return vs v-function
    y = R - V
    K = 7 # Knumber of Knearest Kneighbors
    knn = neighbors.KNeighborsRegressor(K,weights='distance')

    bounds = problem.gen_model.boundary.boundary
    cuts = [np.linspace(l,u,N) for (l,u) in bounds]
    X,Y = np.meshgrid(*cuts)
    P = discrete.make_points(cuts)

    Z = knn.fit(states,y).predict(P)
    assert((N*N,) == Z.shape)
    Z = np.reshape(Z,(N,N),order='F')

    g = problem.discount
    plt.pcolor(X,Y,Z)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('V')
    plt.title(args.title)
    plt.savefig(args.img_out_file)
