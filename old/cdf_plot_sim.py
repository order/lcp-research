import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

import utils.plotting

###############
# Entry point #
###############
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'Usage: <save file> <data files...> <> <>'
        quit()
    save_file = sys.argv[1]
    data_files = sys.argv[2:]

    handles = []
    for data_file in data_files:
        FH = open(data_file,'r')
        (V,R,S) = pickle.load(FH)
        FH.close()
        
        (xs,fs) = utils.plotting.cdf_points(R)
        h = plt.plot(xs,fs,label=data_file)
        handles.append(h)
    plt.title('Simulation Returns')
    plt.legend(data_files,loc='best')
    plt.savefig(save_file)        

    

    
