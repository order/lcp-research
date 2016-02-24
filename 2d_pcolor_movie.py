import numpy as np
from argparse import ArgumentParser

import utils

def parse_command_line():
    parser = ArgumentParser(__file__,'Creates a 2d animation for iterative MDP solvers')
    parser.add_argument('data_file',
                        help='filename with numpy data (.npz format)',
                        type=str)
    parser.add_argument('-o','--output',
                        metavar='FILE',
                        help='save file',
                        type=str)
    parser.add_argument('-p','--processors',
                        metavar='CMD',
                        help='processing commands separated by ";"'
                        + ' (see "apply_processor")',
                        default='2d_primal_value_movie',
                        type=str)
    parser.add_argument('-f','--frames',
                        metavar='INT',
                        help='max number of frames',
                        default=100,
                        type=int)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    args = parse_command_line()

    (root_file,_) = utils.split_extension(args.data_file,'npz')
    pickle_file = root_file + '.pickle'

    # Load data
    data = np.load(args.data_file)
    params = utils.processing.read_pickle(pickle_file)

    # Transform data
    command_queue = args.processors.split(';')
    data = utils.processing.apply_command_queue(command_queue,data,params)
    assert(3 == len(data.shape))
    (I,X,Y) = data.shape

    # Animate the data
    plot_args = utils.kwargify()
    if hasattr(args,'save_file') and args.save_file:
        plot_args['save_file'] = args.save_file

    # Frame skip to maintain max frames
    num_frames = args.frames
    fs = int(np.floor(max(1.0,float(I) / num_frames)))
    data = data[::fs,:,:]
    print data.shape
    assert(data.shape[0] <= num_frames+1)
    
    utils.plotting.animate_frames(data,**plot_args)

    # WHY NOT JUST USE ANALYZE: Allows better use of defaults, so can mostly
    # just run "python ___.py data/test.npz" without other complicated stuff.
    
    # TODO:
    # 1) Make "analyze.py" a callable thing, so this file's job is just
    # to provide a customized and simplified interface to it.
    # 2) Add title and label stuff.
    # 3) Get it working!!

    
    
    
    
