import sys

import numpy as np
import matplotlib.pyplot as plt

import utils
from utils.parsers import KwargParser,ConfigParser


###############
# Entry point #
###############
if __name__ == '__main__':
    if 5 > len(sys.argv):
        print 'Usage: <data file> <plot object> <save_file>'\
            + '<processing...>'
        quit()
    (_,data_file,plotter_file,save_file) = sys.argv[:4]
    processing_commands = sys.argv[4:]

    # Open files
    data_ext = '.npz'
    pickle_ext = '.pickle'
    assert(data_file.endswith(data_ext))
    
    root_file = data_file[:-len(data_ext)] # Strip extension    
    pickle_file = root_file + pickle_ext # Add 'pickle'    
    
    # Load data
    data = np.load(data_file)
    params = utils.processing.read_pickle(pickle_file)

    # Process the data
    data = utils.processing.apply_command_queue(processing_commands,
                                                data,
                                                params)

    # Display it
    plotter = utils.get_instance_from_file(plotter_file)
    print 'Plotting using', plotter
    plotter.display(data,save_file)
