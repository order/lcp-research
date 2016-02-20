import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import utils

def parse_command_line():
    parser = ArgumentParser(__file__,'Creates a 2d animation for iterative MDP solvers')
    parser.add_argument('sim_file',
                        help='file with simulation results (npz format)',
                        type=str)
    parser.add_argument('-o','--output',
                        metavar='FILE',
                        help='save file',
                        type=str)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    args = parse_command_line()

    # Load data
    sim_data = np.load(args.sim_file)

    X = sim_data['states']
    
    (R,I,D) = X.shape
    assert(D == 2) # Velocity and position

    for r in xrange(R):
        plt.plot(X[r,:,0],X[r,:,1],'-k',alpha=0.25)
    #for r in xrange(R):
    #    plt.scatter(X[r,:,0],X[r,:,1],
    #                edgecolor='none')
    plt.show()

    
    
    
    
