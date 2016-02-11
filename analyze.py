import numpy as np
import utils
from utils.parsers import KwargParser,ConfigParser

import matplotlib.pyplot as plt
import time
import sys
import pickle
import types
import os

ProcessorShortcuts = {'2d_value_movie':['primal',
                                        'toframes',
                                        'x[:,0,:,:]'],
                      '2d_flow_movie':['primal',
                                       'toframes',
                                       'np.sum(x[:,1:,:,:],axis=1)'],
                      '2d_adv_movie':['primal',
                                      'toframes',
                                      'advantage'],
                      '2d_policy_movie':['primal',
                                         'toframes',
                                         'policy'],
                      '1d_final_value':['primal',
                                        'toframes',
                                        'x[-1,0,:]'],
                      '1d_final_flows':['primal',
                                        'toframes',
                                        'x[-1,1:,:].T']
                      
}

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

def apply_processor(keyword,x,params):
    if keyword.endswith('.py'):
        filename = keyword
    else:
        filename = 'config/processor/' + keyword + '.py'
        
    if os.path.isfile(filename):
        print '\tUsing file',filename
        processor = utils.get_instance_from_file(filename)
        return processor.process(x,**params)

    if keyword in x:
        print '\tUsing data[{0}]'.format(keyword)
        return x[keyword]

    if keyword in np.__dict__:
        print '\tUsing np.{0}(data)'.format(keyword)
        return np.__dict__[keyword](x)

    print '\tEvaluating',keyword
    return eval('{0}'.format(keyword))

###############
# Entry point #
###############
if __name__ == '__main__':
    if 5 > len(sys.argv):
        print 'Usage: <data file> <plot config file> <save_file>'\
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
    params = read_pickle(pickle_file)

    # Process the data
    iter_count = 0
    while len(processing_commands) > 0:
        iter_count += 1
        assert(iter_count < 50) # Inf loop check.
        
        command = processing_commands.pop(0)
        print 'Command',command
        if command in ProcessorShortcuts:
            # Expand the shortcut
            processing_commands = ProcessorShortcuts[command] + processing_commands
        else:
            # Apply the processor and continue
            data = apply_processor(command,data,params)

    # Display it
    plotter = utils.get_instance_from_file(plotter_file)
    plotter.display(data,save_file)
