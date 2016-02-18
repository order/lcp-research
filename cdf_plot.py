import sys

import numpy as np
import matplotlib.pyplot as plt

import utils.plotting

###############
# Entry point #
###############
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: <data files...> <> <>'
        quit()
    data_files = sys.argv[1:]

    for data_file in data_files:
        assert(data_file.endswith('.npy'))
        data = np.load(data_file)
        (xs,fs) = utils.plotting.cdf_points(data)
        plt.plot(xs,fs)
    plt.show()
        

    

    
