import numpy as np
import utils
from utils.parsers import KwargParser

import matplotlib.pyplot as plt
import time
from optparse import OptionParser
import pickle

def read_pickle(filename):
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
    
    root_filename = data_file[:-len(data_ext)]    
    pickle_filename = root_filename + pickle_ext
    
    print 'Source data file:',data_filename
    print 'Source pickle file:',pickle_filename

    # Load configuration
    parser = ConfigParser(plot_conf_file)
    parser.add_handler('data_process_fn',load_class)
    parser.add_handler('plotting_fn',load_class)
    args = parser.parse()

    # filter out data processing args

    hier_args = util.parsers.hier_key_dict(args,'.')
    data_opts = hier_args.get('data',{})
    plot_opts = hier_args.get('plot',{})
    hier_args.pop('data',None)
    hier_args.pop('plot',None)
    assert(0 == len(hier_args)) # only top-level options

    # FROM HERE....
    
    # Load data
    data = np.load(data_filename)
    params = read_pickle(pickle_filename)

    # Set funcs (Hardcoded for now)
    #process_fn = utils.processing.frame_processor
    #plot_fn = utils.plotting.animate_frames
    
    plot_command = eval(args['command_fn_str'])
    plot_fn = utils.plotting.animate_cdf

    command_fn = 
    
    # Filter out other options (Hardcoded for now)

    title = '{0} with {1}, {2} {3}'.format(params['inst_name'],
                                           params['solver_name'],
                                           args['data_field'].capitalize(),
                                           args['process_fn_str'])
    plotting_options = {'title':title}
    if 'save_file' in args:
        plotting_options['save_file'] = args['save_file']
    processing_options = {'fn' : args['process_fn_str'],
                          'field' : args['data_field']}
   
    processed_data = process_fn(data,params,**processing_options)
    plot_fn(processed_data,**plotting_options)
