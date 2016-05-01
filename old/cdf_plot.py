import sys

import numpy as np
import matplotlib.pyplot as plt

import utils.plotting

###############
# Entry point #
###############
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'Usage: <field> <data files...> <> <>'
        quit()
    field = sys.argv[1]
    data_files = sys.argv[2:]

    handles = []
    for data_file in data_files:
        assert(data_file.endswith('.npz'))
        data = np.load(data_file)[field]
        (xs,fs) = utils.plotting.cdf_points(data)
        h = plt.plot(xs,fs,label=data_file)
        handles.append(h)
    plt.title(field)
    plt.legend(data_files)
    plt.show()
        

    

    
