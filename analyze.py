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
    parser.add('discretizer',None)
    parser.add_optional('solver')
    parser.add_optional('objects')
    parser.add('solver_name','Unknown solver')
    parser.add('inst_name','Unknown instance')
    params = parser.parse(params)

    discretizer = params['discretizer']
    assert(2 == discretizer.get_dimension())    
    return params

def read_config_file(filename):
    """
    Read in the analysis configuration file.
    Should have details about the data processing function
    and the plotting function.
    """

    data_process_str = 'data_process_fn'
    data_prefix = 'data'
    plotting_str = 'plotting_fn'
    plot_prefix = 'plot'
    
    # Load configuration
    parser = ConfigParser(filename)
    parser.add_handler(data_process_str,utils.load_class)
    parser.add_handler(plotting_str,utils.load_class)
    args = parser.parse()

    # filter out data processing args
    data_process_fn = args[data_process_str]
    plotting_fn = args[plotting_str]
    assert(isinstance(data_process_fn,types.FunctionType))
    assert(isinstance(plotting_fn,types.FunctionType))
    del args[data_process_str]
    del args[plotting_str]

    hier_args = utils.parsers.hier_key_dict(args,':')
    data_opts = hier_args.pop(data_prefix,{})
    plot_opts = hier_args.pop(plot_prefix,{})
    assert(0 == len(hier_args)) # only top-level options

    return (data_process_fn,data_opts,plotting_fn,plot_opts)
    
if __name__ == '__main__':
    ## Parser command line

    if 4 != len(sys.argv):
        print 'Usage: python {0} <data file> <plot config file> <save_file>'\
            .format(sys.argv[0])
        quit()
    (_,data_file, plot_conf_file,save_file) = sys.argv

    # Open files
    data_ext = '.npz'
    pickle_ext = '.pickle'
    assert(data_file.endswith(data_ext))
    
    root_file = data_file[:-len(data_ext)]    
    pickle_file = root_file + pickle_ext
    
    print 'Source data file:',data_file
    print 'Source pickle file:',pickle_file

    # Read the data processing and plotting command
    # configuration here
    (data_fn,data_opts,plot_fn,plot_opts)\
        = read_config_file(plot_conf_file)
    
    # Load data
    data = np.load(data_file)
    params = read_pickle(pickle_file)

    # Process the data
    processed_data = data_fn(data,params,**data_opts)

    # Plot the processed data
    plot_opts['save_file'] = save_file
    plot_fn(processed_data,**plot_opts)
