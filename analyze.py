import numpy as np
import utils
from utils.parsers import KwargParser,ConfigParser

import matplotlib.pyplot as plt
import time,sys,pickle,types

def read_pickle(filename):
    """
    Read in information from a pickle file
    Should have the discretizer, at least
    """
    params = pickle.load(open(filename,'rb'))
    parser = KwargParser()
    parser.add('instance_builder')
    parser.add('solver_generator')
    parser.add('inst_conf_file')
    parser.add('solver_conf_file')
    parser.add('objects')
    params = parser.parse(params)   
    return params



###############
# Entry point #
###############
if __name__ == '__main__':

    if 5 != len(sys.argv):
        print 'Usage: <data file> <processor file> <plot config file> <save_file>'\
            .format(sys.argv[0])
        quit()
    (_,data_file, processor_file, plotter_file,save_file) = sys.argv

    # Open files
    data_ext = '.npz'
    pickle_ext = '.pickle'
    assert(data_file.endswith(data_ext))
    
    root_file = data_file[:-len(data_ext)] # Strip extension    
    pickle_file = root_file + pickle_ext # Add 'pickle'    
    
    # Load data
    data = np.load(data_file)
    params = read_pickle(pickle_file)

    # Process the data
    processor = utils.get_instance_from_file(processor_file)
    processed_data = processor.process(data)

    # Display it
    plotter = utils.get_instance_from_file(plotter_file)
    plotter.display(processed_data,save_file)
