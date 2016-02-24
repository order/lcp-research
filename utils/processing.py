import numpy as np
import pickle
import os

import utils
from parsers import KwargParser

ProcessorShortcuts = {'2d_primal_value_movie':['primal',
                                        'toframes',
                                        'x[:,0,:,:]'],
                      '2d_value_movie':['value',
                                        'toframesvalue'],
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
                                        'final',
                                        'toframes',
                                        'x[-1,0,:]'],
                      '1d_final_flows':['primal',
                                        'final',
                                        'toframes',
                                        'x[-1,1:,:].T'],
                      'final':['(x[-1,:])[np.newaxis,:]']

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

def apply_command_queue(command_queue,data,params):
    iter_count = 0
    command_queue = list(reversed(command_queue))
    while len(command_queue) > 0:
        iter_count += 1
        assert(iter_count < 30) # Anti-inf loop kludge

        command = command_queue.pop()
        if command in ProcessorShortcuts:
            # Expand the shortcut
            reversed_command = reversed(ProcessorShortcuts[command])
            command_queue.extend(reversed_command)
        else:
            data = apply_single_processor(command,data,params)
    return data        

def apply_single_processor(keyword,x,params):
    # First priority: is a file name in config/processor
    if keyword.endswith('.py'):
        filename = keyword
    else:
        filename = 'config/processor/' + keyword + '.py'
    if os.path.isfile(filename):
        print '\tUsing file',filename
        processor = utils.get_instance_from_file(filename)
        return processor.process(x,**params)

    # Secord priority: is a key in the data
    if keyword in x:
        print '\tUsing data[{0}]'.format(keyword)
        return x[keyword]

    # Third priority: just evaluate it
    print '\tEvaluating',keyword
    return eval('{0}'.format(keyword))
            

def split_into_frames(Data,A,n,lens):
    """
    Take a (I,N) matrix where the columns are either
    primal or dual block-structured vectors from an 
    LCP'd MDP. The MDP should be based on 2 continuous
    dimensions (e.g. the double integrator)

    Convert into an (I,(A+1),X1,X2,...) tensor
    """
    (I,N) = Data.shape
    assert(N == (A+1)*n)
    xyz = np.prod(lens)
    assert(xyz <= n )

    # Reshape into I, n, (A+1)
    Frames = np.reshape(Data,(I,n,(A+1)),order='F')
    
    # Crop out non-physical states
    Frames = Frames[:,:xyz,:]
    
    # Reshape into  I, X1, X2, X3, (A+1)
    dims = [I] + lens+ [A+1]
    Frames = np.reshape(Frames,dims,order='C')
    
    # Swap axes to be I x (A+1) x Y x X
    Frames = np.rollaxis(Frames,-1,1)
    return Frames

def final_frame(data,params,**kwargs):
    """
    Like frame processing, but only the final frame
    """
    
    # Inner functions for different processing options
    def value(F):
        return F[0,:,:]

    def flow(F):
        return np.argmax(F[1:,:,:],axis=0)

    def agg_flow(F):
        return np.sum(F[1:,:,:],axis=0)

    def log_agg_flow(F):
        return np.log10(np.sum(F[1:,:,:],axis=0))

    def adv(F):
        SF = np.sort(F[1:,:,:],axis=0)
        return np.log(SF[-1,:,:] - SF[-2,:,:] + 1e-22)

    parser = KwargParser()
    parser.add('field','primal') #primal/dual
    parser.add('option','value')
    args = parser.parse(kwargs)

    if 'comp' == args['field']:
        data_field = data['primal'] * data['dual']
    else:
        data_field = data[args['field']]
    fn = eval(args['option'])
    
    discretizer = params['discretizer']
    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()
    (x,y) = discretizer.get_basic_lengths()    
    frames = split_into_frames(data_field,A,n,x,y)

    return fn(frames[-1,:,:,:])
    

def frame_processing(data,params,**kwargs):
    """
    Turns data into movie frames based on provided inputs
    """

    # Inner functions for different processing options
    def value(F):
        return F[:,0,:,:]

    def flow(F):
        return np.argmax(F[:,1:,:,:],axis=1)

    def agg_flow(F):
        return np.sum(F[:,1:,:,:],axis=0)

    def adv(F):
        SF = np.sort(F[:,1:,:,:],axis=1)
        return np.log(SF[:,-1,:,:] - SF[:,-2,:,:] + 1e-22)

    parser = KwargParser()
    parser.add('field','primal') #primal/dual
    parser.add('option','value')
    args = parser.parse(kwargs)

    if 'comp' == args['field']:
        data_field = data['primal'] * data['dual']
    else:
        data_field = data[args['field']]
    fn = eval(args['option'])
    
    discretizer = params['discretizer']
    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()
    (x,y) = discretizer.get_basic_lengths()    
    frames = split_into_frames(data_field,A,n,x,y)

    return fn(frames)

def vector_processing(data,params,**kwargs):
    """
    Returns a subset of the data based on provided inputs
    """
    def val(F,n):
        return F[:,:n]

    def flow(F,n):
        return F[:,n:]

    def log_flow(F,n):
        return np.log10(flow(F,n))

    parser = KwargParser()
    parser.add('field','primal') #primal/dual
    parser.add('option','val')
    args = parser.parse(kwargs)

    if 'comp' == args['field']:
        data_field = data['primal'] * data['dual']
    else:
        data_field = data[args['field']]
        
    fn = eval(args['option'])
        
    discretizer = params['discretizer']

    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()

    return fn(data_field,n)

def time_series(data,params,**kwargs):
    
    def identity(x):
        return x

    def log(x):
        return np.log10(x)

    def log_vector_cond(X):
        """
        Assumes X > 0, and it's of the form (I,N) = X.shape
        where the rows are iteration numbers and columns are
        components of the solution vector for that iteration.

        The "vector condition number" is the ratio between the
        max and the min values of X.

        Log is taken because the condition numbers can blow up.
        """
        assert (2 == len(X.shape))
        assert(not np.any(X <= 0))
        (I,N) = X.shape
        ret = np.amax(X,axis=1) / np.amin(X,axis=1)
        assert((I,) == ret.shape)
        return np.log10(ret)

    parser = KwargParser()
    parser.add('field')
    parser.add('option','identity')
    args = parser.parse(kwargs)

    fn = eval(args['option'])
    
    ret = fn(data[args['field']])
    assert(1 == len(ret.shape)) # Should be a vector

    return ret
