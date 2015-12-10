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
    parser = OptionParser()
    parser.add_option('-d','--datafile',dest='data_filename',
                      help='data file in .npz format',
                      metavar='FILE')
    parser.add_option('-p','--process',dest='process_fn_str',
                      help='processing function (fn in util.processing)',
                      metavar='FN')
    parser.add_option('-f','--field',dest='data_field',
                      help='data field name (primal,dual,...)',
                      metavar='NAME')
    parser.add_option('-o','--out',dest='save_file',
                      help='save file', metavar = 'FILE')
    (options,_) = parser.parse_args()

    # Command line overrides file settings
    raw_args = {} # TODO read stuff from file
    for (k,v) in options.__dict__.items():
        if v:
            raw_args[k] = v
    print raw_args

    arg_parser = KwargParser()
    arg_parser.add('data_filename','data/test.npz')
    arg_parser.add_optional('pickle_filename')
    arg_parser.add('process_fn_str','val')
    arg_parser.add('data_field','primal')
    arg_parser.add_optional('save_file')
    args = arg_parser.parse(raw_args)

    # Open files
    data_ext = '.npz'
    pickle_ext = '.pickle'
    data_filename = args['data_filename']
    assert(data_filename.endswith(data_ext))
    root_filename = data_filename[:-len(data_ext)]

    
    pickle_filename = args.get('pickle_filename',
                               root_filename + pickle_ext)    
    assert(pickle_filename.endswith(pickle_ext))

    print 'Source data file:',data_filename
    print 'Source pickle file:',pickle_filename
    
    # Load data
    data = np.load(data_filename)
    params = read_pickle(pickle_filename)

    # Set funcs (Hardcoded for now)
    process_fn = utils.processing.frame_processor
    plot_fn = utils.plotting.animate_frames

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
