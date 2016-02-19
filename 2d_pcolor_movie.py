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
                        default='2d_value_movie',
                        type=str)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    args = parse_command_line()

    (root_file,_) = utils.split_extension(args.data_file,'npz')
    pickle_file = root_file + '.pickle'

    # Load data
    data = np.load(data_file)
    params = read_pickle(pickle_file)

    # Transform data
    command_queue = args.processors.split(';')
    data = apply_command_queue(command_queue,data,params)
    assert(3 == len(data.shape))

    # Animate the data
    plot_args = utils.kwargify()
    if args.save_file:
        plot_args['save_file'] = args.save_file
    
    utils.plotting.animate_frames(data,**plot_args)

    # WHY NOT JUST USE ANALYZE: Allows better use of defaults, so can mostly
    # just run "python ___.py data/test.npz" without other complicated stuff.
    
    # TODO:
    # 1) Make "analyze.py" a callable thing, so this file's job is just
    # to provide a customized and simplified interface to it.
    # 2) Add title and label stuff.
    # 3) Get it working!!

    
    
    
    
